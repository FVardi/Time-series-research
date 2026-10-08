"""Probes and metrics on frozen representations (D26).

Two probes:
  svm_ts2vec : TS2Vec's own SVM protocol, run with the official code (reproduction check)
  logreg     : the harness probe, logistic regression with C chosen by cross-validation
"""

import importlib.util
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load TS2Vec's evaluation code straight from its file. Importing it as 'tasks._eval_protocols'
# would also run tasks/__init__.py, which needs extra packages for forecasting.
_path = os.path.join(os.path.dirname(__file__), "..", "external", "ts2vec", "tasks", "_eval_protocols.py")
_spec = importlib.util.spec_from_file_location("ts2vec_eval_protocols", os.path.abspath(_path))
ts2vec_protocols = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts2vec_protocols)


def label_subset(y, fraction, seed):
    """Indices of a stratified random subset holding 'fraction' of the labels (D26 item 5).

    A new subset is drawn for each seed, so the spread over seeds includes
    the effect of which labels were drawn (D29).
    """
    idx = np.arange(len(y))
    if fraction >= 1.0:
        return idx
    subset, _ = train_test_split(idx, train_size=fraction, stratify=y, random_state=seed)
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


PROBES = {"svm_ts2vec": svm_ts2vec, "logreg": logreg}


def scores(clf, Z, y):
    """Accuracy and macro-F1 on the test data (D26 item 3)."""
    pred = clf.predict(Z)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
