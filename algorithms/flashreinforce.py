"""Reference fused-batch semantics; replace with kernels only after profiling."""

from algorithms.reinforce import REINFORCE


class FlashREINFORCE(REINFORCE):
    name = "flashreinforce"
