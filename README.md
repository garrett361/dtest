# Distributed Pytest Utils

`pytest` plugin for distributed `torch` tests. Shamelessly stolen from/inspired by DeepSpeed, then
simplified.

```
uv pip install git+https://github.com/garrett361/dtest
```

Minimal example:

```python
import pytest
import torch
import torch.distributed as dist

from dtest import DTest


class TestDTest(DTest):
    @pytest.mark.parametrize("n", list(range(1, 4)))
    def test_all_reduce(self, n: int) -> None:
        t = torch.arange(n * self.world_size, device=self.device)
        dist.all_reduce(t)
        self.print_rank(f"{t=}")
```

Uses all available GPUs by default, or if on CPU defaults to `world_size=2`, unless the class
attribute `default_world_size` is edited. The world size can also be configured by using the
`world_size` mark _and_ specifying a `world_size` arg:

```python
class TestWorldSizes(DTest):
    def test_default_world_size(self) -> None:
        self.print_rank0_only(f"{self.world_size=}")

    @pytest.mark.world_size(5)
    def test_non_default_world_size(self, world_size: int) -> None:
        self.print_rank0_only(f"{self.world_size=}")

    @pytest.mark.world_size([2, 3, 4])
    def test_multiple_world_sizes(self, world_size: int) -> None:
        self.print_rank0_only(f"{self.world_size=}")
```
See `test.py` for more cases, which also serves as the best documentation.
