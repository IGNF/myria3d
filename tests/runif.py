import pytest
import torch

"""
Simplified from:
    https://github.com/ashleve/lightning-hydra-template/blob/main/tests/helpers/runif.py
    which adapted it from
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/helpers/runif.py
"""


class RunIf:
    """RunIf wrapper for conditional skipping of tests.

    Fully compatible with `@pytest.mark`.
    Example:
        @RunIf(min_gpus=1)
        @pytest.mark.parametrize("arg1", [1.0, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0

    """

    def __new__(
        self,
        min_gpus: int = 0,
        **kwargs,
    ):
        """
        Args:
            min_gpus: min number of gpus required to run test
            kwargs: native pytest.mark.skipif keyword arguments
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )
