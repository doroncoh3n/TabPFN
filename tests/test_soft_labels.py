
import numpy as np
import pytest
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

def test_soft_labels_fit_predict():
    """Test that TabPFNClassifier can fit and predict with soft labels."""
    # Create dummy data
    X = np.random.rand(20, 5)

    # Soft labels: 20 samples, 3 classes
    y_soft = np.random.rand(20, 3)
    y_soft = y_soft / y_soft.sum(axis=1, keepdims=True)

    # Use V2 model which can be downloaded directly
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu')

    # Fit should not raise
    clf.fit(X, y_soft)

    # Predict should work
    y_pred = clf.predict(X)
    assert y_pred.shape == (20,)

    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (20, 3)

def test_soft_labels_validation_split():
    """Test that validation split works with soft labels."""
    X = np.random.rand(100, 5)
    y_soft = np.random.rand(100, 3)
    y_soft = y_soft / y_soft.sum(axis=1, keepdims=True)

    # Enable tuning to trigger validation split logic
    tuning_config = {
        "calibrate_temperature": True,
        "tune_decision_thresholds": True,
        "tuning_holdout_frac": 0.2
    }

    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device='cpu',
        tuning_config=tuning_config,
        eval_metric="accuracy"
    )

    # This should trigger get_tuning_splits and find_optimal_classification_thresholds
    clf.fit(X, y_soft)

    assert clf.tuned_classification_thresholds_ is not None
    assert clf.softmax_temperature_ is not None

def test_soft_labels_permutation():
    """Test that class permutation works correctly with soft labels."""
    # This is a bit harder to test end-to-end without inspecting internals,
    # but we can check if it runs without error.
    X = np.random.rand(10, 5)
    y_soft = np.random.rand(10, 3)
    y_soft = y_soft / y_soft.sum(axis=1, keepdims=True)

    # n_estimators=2 will likely trigger some class permutation
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device='cpu',
        n_estimators=2
    )

    clf.fit(X, y_soft)
    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (10, 3)
