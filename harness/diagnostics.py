"""Collapse diagnostics for learned representations (D71, D73).

Every function takes a representation matrix Z of shape (n windows, d dimensions) as a
torch tensor and only measures it; nothing here is used in a loss. All measures are
computed in float64, so tiny eigenvalues are not lost to rounding.

    rankme                 effective rank (RankMe, Garrido et al. 2023, arXiv 2210.02885)
    covariance_eigenvalues eigenvalue spectrum of the covariance of Z
    dimension_std          per-dimension standard deviation (mean, and fraction of 'dead' dimensions)
    mean_pairwise_cosine   mean cosine similarity between different windows
    alignment              mean cosine similarity between each window and its augmented version
    sigreg                 SIGReg statistic (LeJEPA, Balestriero & LeCun 2025, arXiv 2511.08544)

Complete collapse: all windows map to the same point (std 0, cosine 1).
Dimensional collapse: the windows still differ, but only along a few directions
(small effective rank, most eigenvalues near 0).
"""

import math

import torch
import torch.nn.functional as F


def rankme(Z, epsilon):
    """RankMe (paper, eq. 1-2): exp(-sum_k p_k log p_k), p_k = sigma_k(Z) / ||sigma(Z)||_1 + epsilon.

    sigma(Z) are the singular values of Z itself, as written in the paper (it does not say
    whether Z is centred, so it is not centred here; the covariance spectrum below is).
    The paper gives no value for epsilon; it comes from the config.
    Between 1 (one direction) and min(n, d) (all singular values equal).
    """
    s = torch.linalg.svdvals(Z.double())
    p = s / s.sum() + epsilon
    return float(torch.exp(-(p * torch.log(p)).sum()))


def covariance_eigenvalues(Z):
    """Eigenvalues of the covariance matrix of Z (windows centred, divided by n - 1), largest first.

    Computed as squared singular values of the centred Z, which is cheaper and more accurate
    than forming the d x d covariance. There are min(n, d) of them; with n <= d at most n - 1
    can be non-zero, whatever the encoder does, so n should be well above d.
    """
    Zc = Z.double() - Z.double().mean(dim=0)
    return torch.linalg.svdvals(Zc) ** 2 / (Z.shape[0] - 1)


def dimension_std(Z, relative_threshold):
    """Standard deviation of every dimension over the windows (n - 1 in the denominator).

    Returns its mean and the fraction of 'dead' dimensions: those whose standard deviation is
    below relative_threshold times the largest standard deviation of this representation (D74;
    relative, because the representations differ widely in scale). If every dimension has
    standard deviation 0 (complete collapse), all dimensions count as dead.
    """
    std = Z.double().std(dim=0, correction=1)
    if std.max() == 0:
        return 0.0, 1.0
    return float(std.mean()), float((std < relative_threshold * std.max()).double().mean())


def mean_pairwise_cosine(Z):
    """Mean cosine similarity over all pairs of different windows (1 = all point the same way).

    Uses sum_{i != j} u_i . u_j = ||sum_i u_i||^2 - n for unit vectors u_i, so no n x n matrix
    is formed. A window whose representation is exactly the zero vector has no direction;
    it is counted with cosine 0 (F.normalize leaves it at zero; eps written out below).
    """
    n = Z.shape[0]
    u = F.normalize(Z.double(), dim=1, eps=1e-12)
    return float((u.sum(dim=0).square().sum() - (u.square().sum())) / (n * (n - 1)))


def alignment(Z, Z_aug):
    """Mean cosine similarity between row i of Z and row i of Z_aug (each window vs its augmented version)."""
    return float(F.cosine_similarity(Z.double(), Z_aug.double(), dim=1, eps=1e-12).mean())


def sigreg_directions(d, n_slices, generator):
    """n_slices random unit directions in R^d (columns), Gaussian then normalised, as the official code."""
    A = torch.randn(d, n_slices, generator=generator, device=generator.device, dtype=torch.float64)
    return A / A.norm(p=2, dim=0)


def sigreg(Z, A, t_max, n_points):
    """SIGReg statistic: Epps-Pulley test of each 1-D projection Z @ A against N(0, 1), mean over slices.

    Follows the official LeJEPA code (github.com/rbalestr-lab/lejepa, commit c293d29:
    lejepa/univariate/epps_pulley.py, class EppsPulley, integration 'trapezoid';
    lejepa/multivariate/slicing.py, reduction 'mean'):
        T = N * integral |phi_emp(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt,
    integrated over t in [0, t_max] with the trapezoid rule and doubled (the integrand is
    symmetric in t). 0 for an exact standard Gaussian; grows with N, so values are only
    comparable for the same number of windows. Not scale-free: Z is not standardised,
    because matching N(0, I) includes the scale.
    """
    N = Z.shape[0]
    t = torch.linspace(0, t_max, n_points, dtype=torch.float64, device=Z.device)
    dt = t_max / (n_points - 1)
    weights = torch.full((n_points,), 2 * dt, dtype=torch.float64, device=Z.device)
    weights[[0, -1]] = dt  # trapezoid end points; the factor 2 accounts for t < 0
    phi = torch.exp(-t.square() / 2)  # characteristic function of N(0, 1)
    x_t = (Z.double() @ A).unsqueeze(-1) * t  # (N, slices, points)
    err = (x_t.cos().mean(0) - phi).square() + x_t.sin().mean(0).square()  # (slices, points)
    return float(((err @ (weights * phi)) * N).mean())


def measures(Z, s, A=None, Z_aug=None):
    """All scalar measures of one representation, plus its covariance eigenvalues.

    s: the 'diagnostics' settings (rankme_epsilon, std_threshold_relative, sigreg.t_max, sigreg.n_points).
    A: SIGReg directions for this representation's dimension; Z_aug: augmented version, or None.
    If Z contains NaN or inf (diverged training), every measure is NaN.
    """
    row = {"n": Z.shape[0], "d": Z.shape[1], "finite": bool(torch.isfinite(Z).all())}
    names = ["rankme", "mean_std", "frac_dead", "mean_norm", "mean_cosine", "sigreg", "alignment"]
    if not row["finite"]:
        return {**row, **{k: math.nan for k in names}}, None
    eig = covariance_eigenvalues(Z)
    row["rankme"] = rankme(Z, s["rankme_epsilon"])
    row["mean_std"], row["frac_dead"] = dimension_std(Z, s["std_threshold_relative"])
    row["mean_norm"] = float(Z.double().norm(dim=1).mean())  # scale of the representation
    row["mean_cosine"] = mean_pairwise_cosine(Z)
    row["sigreg"] = sigreg(Z, A, s["sigreg"]["t_max"], s["sigreg"]["n_points"])
    row["alignment"] = alignment(Z, Z_aug) if Z_aug is not None else math.nan
    return row, eig.cpu().numpy()
