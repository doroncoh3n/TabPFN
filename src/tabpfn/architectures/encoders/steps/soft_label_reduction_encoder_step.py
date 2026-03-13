"""Encoder step to reduce soft labels to a scalar expected value."""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import torch

from tabpfn.architectures.encoders import TorchPreprocessingStep


class SoftLabelReductionEncoderStep(TorchPreprocessingStep):
    """Encoder step that reduces soft labels (probabilities) to a scalar expected index.

    This effectively computes sum(p_i * i) for each sample, exploiting the linearity
    of the subsequent LinearInputEncoderStep to produce a weighted average embedding.
    """

    def __init__(
        self,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
        nan_keys: tuple[str, ...] = ("nan_indicators",),
    ):
        # We modify nan_keys as well, so we must declare them in out_keys
        # to pass validation in TorchPreprocessingStep.
        all_out_keys = list(out_keys)
        for k in nan_keys:
            if k not in all_out_keys:
                all_out_keys.append(k)

        super().__init__(in_keys, tuple(all_out_keys))
        self.nan_keys = nan_keys
        # We need to know which keys are "main" outputs corresponding to in_keys
        self.main_out_keys = out_keys

    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """No fitting required."""
        pass

    @override
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Reduce soft labels to scalar."""
        outputs = {}

        # Process main keys (the soft labels)
        for in_k, out_k in zip(self.in_keys, self.main_out_keys):
            x = state[in_k]
            # x shape: [seq, batch, n_classes]
            # We want to reduce last dim

            n_classes = x.shape[-1]
            if n_classes <= 1:
                # Already scalar?
                outputs[out_k] = x
                continue

            indices = torch.arange(n_classes, device=x.device, dtype=x.dtype)
            # sum(p_i * i)
            # [seq, batch, n_classes] * [n_classes] -> [seq, batch, n_classes] -> sum -> [seq, batch]
            scalar_y = (x * indices).sum(dim=-1, keepdim=True)
            outputs[out_k] = scalar_y

        # Process nan keys (reduce them to size 1)
        # We assume nan keys are present in state but not in self.in_keys/out_keys
        # usually passed separately or we need to know their names.
        # NanHandlingEncoderStep produces "nan_indicators".

        for k in self.nan_keys:
            if k in state:
                nan_ind = state[k]
                if nan_ind.shape[-1] > 1:
                    # If any class is nan, we treat the scalar as nan
                    outputs[k] = nan_ind.any(dim=-1, keepdim=True).float()
                else:
                    outputs[k] = nan_ind

        return outputs
