from __future__ import annotations

import numpy as np

from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig
from tabpfn.preprocessing.transform import _transform_labels_one


def _make_classifier_config(class_permutation: np.ndarray | None) -> ClassifierEnsembleConfig:
    return ClassifierEnsembleConfig(
        preprocess_config=PreprocessorConfig(name="none"),
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_count=0,
        feature_shift_decoder=None,
        subsample_ix=None,
        outlier_removal_std=None,
        _model_index=0,
        class_permutation=class_permutation,
    )


def test__transform_labels_one__samples_from_soft_labels() -> None:
    y_soft = np.array(
        [
            [0.6, 0.4],
            [0.6, 0.4],
            [0.6, 0.4],
            [0.6, 0.4],
        ]
    )
    cfg = _make_classifier_config(class_permutation=None)

    sampled = _transform_labels_one(cfg, y_soft, random_state=0)

    # If we converted to hard labels via argmax, this would be all zeros.
    assert not np.array_equal(sampled, np.zeros(len(y_soft), dtype=np.int64))


def test__transform_labels_one__applies_permutation_after_soft_label_sampling() -> None:
    y_soft = np.array(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ]
    )
    permutation = np.array([1, 0])
    cfg = _make_classifier_config(class_permutation=permutation)

    sampled_without_perm = _transform_labels_one(
        _make_classifier_config(class_permutation=None),
        y_soft,
        random_state=4,
    )
    sampled_with_perm = _transform_labels_one(cfg, y_soft, random_state=4)

    np.testing.assert_array_equal(sampled_with_perm, permutation[sampled_without_perm])
