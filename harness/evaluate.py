"""Probes and metrics on frozen representations (D26).

Probes:
  svm_ts2vec       : TS2Vec's own SVM protocol, run with the official code (reproduction check)
  logreg           : the harness probe, logistic regression with C chosen by cross-validation
  ridge_minirocket : MiniRocket's own ridge classifier (reproduction check for MiniRocket, D81)
  floor            : the most frequent training class, ignoring the features (D83)
Regression (kappa on Herning, D58, D170; XJTU-SY remaining useful life, D232):
  ridge            : ridge regression with alpha chosen by cross-validation
  ridge_grouped    : the same, with folds grouped by bearing (XJTU-SY, D232 item 5)
  floor_mean       : the mean of the training targets, ignoring the features
XJTU-SY (D231, D232): time_window_embedding (DSCN Eq. 16 on the representations) and rul_scores
(DSCN's score and RMSE in minutes, plus MAE and R², over the second half of each test bearing's life).
"""

import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load TS2Vec's evaluation code straight from its file. Importing it as 'tasks._eval_protocols'
# would also run tasks/__init__.py, which needs extra packages for forecasting.
_path = os.path.join(os.path.dirname(__file__), "..", "external", "ts2vec", "tasks", "_eval_protocols.py")
_spec = importlib.util.spec_from_file_location("ts2vec_eval_protocols", os.path.abspath(_path))
ts2vec_protocols = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts2vec_protocols)


def label_subset(y, fraction, seed, stratify=True):
    """Indices of a random subset holding 'fraction' of the labels (D26 item 5), stratified by
    class; unstratified for a numeric target (stratify=False; regression, D170).

    A new subset is drawn for each seed, so the spread over seeds includes
    the effect of which labels were drawn (D29).
    """
    idx = np.arange(len(y))
    if fraction >= 1.0:
        return idx
    subset, _ = train_test_split(idx, train_size=fraction, stratify=y if stratify else None, random_state=seed)
    return np.sort(subset)


def svm_ts2vec(Z, y):
    """TS2Vec's protocol: RBF SVM, C by 5-fold cross-validation over 1e-4..1e4 and inf (official code)."""
    return ts2vec_protocols.fit_svm(Z, y)


def logreg(Z, y):
    """Harness probe: standardise features (training statistics), L2 logistic regression,
    C by 5-fold cross-validation on the training data over 1e-4..1e4.

    Cross-validation scores with accuracy (sklearn's default for classifiers).
    max_iter is set high so the solver converges; the optimum does not depend on it.
    """
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10_000))
    grid = {"logisticregression__C": [10.0**k for k in range(-4, 5)]}
    return GridSearchCV(pipe, grid, cv=5).fit(Z, y).best_estimator_


def grouped_folds(y, groups, seed, n_splits=5, max_draws=100):
    """Cross-validation folds grouped by 'groups' (e.g. bearing) and stratified by class,
    shuffled with the run's seed (D129).

    Guard: if a fold's training part lacks a class, the assignment is drawn again with seed
    seed * 1000 + draw, draw = 1, 2, ... The guard covers only classes whose windows come from
    at least two groups: a class with a single group (e.g. all labelled windows of one bearing
    at a small label budget) is necessarily missing from the training part of the fold that
    holds that group out, and this is accepted (D148).
    Returns the folds, the draw used (0 = none redrawn) and the classes with a single group.
    """
    n_groups = {c: len(set(groups[y == c])) for c in set(y)}
    single = sorted(c for c, n in n_groups.items() if n == 1)
    required = {c for c, n in n_groups.items() if n >= 2}
    for draw in range(max_draws):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed * 1000 + draw)
        folds = list(cv.split(np.zeros(len(y)), y, groups))
        if all(required <= set(y[train]) for train, _ in folds):
            return folds, draw, single
    raise ValueError(f"no fold assignment with every class in every training part after {max_draws} draws")


def logreg_grouped(Z, y, groups, seed):
    """The harness probe (logreg) with C chosen on folds grouped by bearing (D129, D148).
    Same pipeline and grid as logreg(). Returns the probe, the fold draw used and the
    classes whose labelled windows all come from one group."""
    folds, draw, single = grouped_folds(y, groups, seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10_000))
    grid = {"logisticregression__C": [10.0**k for k in range(-4, 5)]}
    return GridSearchCV(pipe, grid, cv=folds).fit(Z, y).best_estimator_, draw, single


def ridge_minirocket(Z, y):
    """MiniRocket's own probe (D81), as in its official README:
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True).

    scikit-learn has removed 'normalize'. Its documented replacement gives the same model:
    standardise the features (training statistics) and multiply every alpha by the number
    of training series. (normalize=True divided each centred feature by its L2 norm, which is
    sqrt(n) times its standard deviation; tests/test_baselines.py checks the equivalence.)
    """
    alphas = np.logspace(-3, 3, 10) * len(y)
    return make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas)).fit(Z, y)


def floor(Z, y):
    """The floor (D75 a, D83): always predicts the most frequent class in the (labelled) training data."""
    return DummyClassifier(strategy="most_frequent").fit(Z, y)


def ridge(Z, y, seed):
    """Regression probe (D58, D59): standardise features (training statistics), ridge regression,
    alpha by 5-fold cross-validation on the training data over 1e-4..1e4 (powers of ten).

    For the Herning validation (D170) the folds are not grouped (D59 groups by hold): shuffled
    with the run's seed, scored with R² (sklearn's default for regressors).
    """
    pipe = make_pipeline(StandardScaler(), Ridge())
    grid = {"ridge__alpha": [10.0**k for k in range(-4, 5)]}
    return GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=seed)).fit(Z, y).best_estimator_


def ridge_grouped(Z, y, groups, seed, n_splits):
    """ridge() with alpha chosen on folds grouped by 'groups' (bearing), shuffled with the run's seed
    (XJTU-SY: 4 folds as DSCN's 4-fold cross-validation, grouped by bearing as D129; D232 item 5)."""
    pipe = make_pipeline(StandardScaler(), Ridge())
    grid = {"ridge__alpha": [10.0**k for k in range(-4, 5)]}
    folds = list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Z, y, groups))
    return GridSearchCV(pipe, grid, cv=folds).fit(Z, y).best_estimator_


def floor_mean(Z, y):
    """Regression floor (as D83 for classes): always predicts the mean of the (labelled) training targets."""
    return DummyRegressor(strategy="mean").fit(Z, y)


PROBES = {"svm_ts2vec": svm_ts2vec, "logreg": logreg, "ridge_minirocket": ridge_minirocket, "floor": floor,
          "floor_mean": floor_mean}  # ridge takes the seed as well; called directly by scripts/evaluate.py


def scores(clf, Z, y):
    """Accuracy and macro-F1 on the test data (D26 item 3)."""
    pred = clf.predict(Z)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}


def regression_scores(reg, Z, y):
    """MAE and R² on the test data (D59)."""
    pred = reg.predict(Z)
    return {"mae": mean_absolute_error(y, pred), "r2": r2_score(y, pred)}


def per_class(clf, Z, y, class_names):
    """Test accuracy per class (D168, D179), a diagnostic next to scores(): for each class, its number
    of test windows and the share of them the probe predicts correctly."""
    pred = clf.predict(Z)
    return [{"class": class_names[k], "n_windows": int((y == k).sum()), "accuracy": float((pred[y == k] == k).mean())}
            for k in sorted(set(y))]


def per_bearing(clf, Z, y, groups, class_names):
    """Test accuracy per test bearing (D160), a diagnostic next to scores().

    One row per group (bearing): its class, number of windows, accuracy and the class
    the probe predicts most often for it."""
    pred = clf.predict(Z)
    rows = []
    for g in sorted(set(groups)):
        m = groups == g
        top = np.bincount(pred[m], minlength=len(class_names)).argmax()  # most frequent prediction
        rows.append({"bearing": g, "class": class_names[y[m][0]], "n_windows": int(m.sum()),
                     "accuracy": float((pred[m] == y[m]).mean()), "top_prediction": class_names[top]})
    return rows


def time_window_embedding(Z, runs, minutes, S):
    """DSCN's time-window embedding (Wang et al., 2019, Eq. 16) applied to frozen representations
    (D232 item 4): the input of minute n is [z_(n-S+1); ...; z_n] of the same run, in time order.

    Z: (n_windows, d), one row per minute; runs, minutes: run ID and minute of each row.
    Rows without S - 1 earlier minutes in their run are left out. Stops if a run has gaps.
    Returns the embedded representations (n_kept, S * d) and the indices of the kept rows in Z.
    """
    runs, minutes = np.asarray(runs), np.asarray(minutes)
    out, kept = [], []
    for r in pd.unique(runs):
        idx = np.flatnonzero(runs == r)
        idx = idx[np.argsort(minutes[idx])]  # time order
        if (np.diff(minutes[idx]) != 1).any():
            raise ValueError(f"run {r}: minutes are not consecutive")
        for j in range(S - 1, len(idx)):
            out.append(Z[idx[j - S + 1: j + 1]].reshape(-1))  # oldest first, current last (Eq. 16)
            kept.append(idx[j])
    return np.stack(out), np.array(kept)


CLIP = (0.0, 0.99)  # predicted normalised RUL before Eq. 15 (D232 item 6)


def phm08_score(d):
    """PHM 2008 scoring function as in DSCN (Eq. 12), d = predicted - actual RUL (minutes):
    sum of exp(-d/13) - 1 for d < 0 and exp(d/10) - 1 for d >= 0."""
    d = np.asarray(d, dtype=np.float64)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rul_scores(pred, y, minutes, lifetime):
    """Remaining-useful-life metrics of one test bearing (D231 item 4, D232 item 6).

    pred, y: predicted and actual normalised RUL ((N - n) / N, D228) of its minutes; minutes: n;
    lifetime: N (minutes). Only minutes n >= N / 2 are scored ("from the half to the end of lifetime").
    For score and RMSE, the prediction is clipped to CLIP and converted to minutes with DSCN's Eq. 15,
    RUL = q * n / (1 - q); the actual RUL in minutes is N - n. MAE and R² are on the normalised
    target, unclipped predictions.
    """
    pred, y, minutes = (np.asarray(a, dtype=np.float64) for a in (pred, y, minutes))
    m = minutes >= lifetime / 2
    q = np.clip(pred[m], *CLIP)
    pre_min = q * minutes[m] / (1 - q)  # Eq. 15
    act_min = lifetime - minutes[m]
    d = pre_min - act_min
    return {"score": phm08_score(d), "rmse_min": float(np.sqrt(np.mean(d**2))),
            "mae": mean_absolute_error(y[m], pred[m]), "r2": r2_score(y[m], pred[m]),
            "n_scored": int(m.sum()), "n_clipped": int(((pred[m] < CLIP[0]) | (pred[m] > CLIP[1])).sum())}
