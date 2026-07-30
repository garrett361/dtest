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
`world_size` mark, which generates a separate test instance per value:

```python
class TestWorldSizes(DTest):
    def test_default_world_size(self) -> None:
        self.print_rank0_only(f"{self.world_size=}")

    @pytest.mark.world_size(5)
    def test_non_default_world_size(self) -> None:
        self.print_rank0_only(f"{self.world_size=}")

    @pytest.mark.world_size([2, 3, 4])
    def test_multiple_world_sizes(self) -> None:
        self.print_rank0_only(f"{self.world_size=}")
```

A `world_size` arg is optional (it was previously required), and still supported if the test
body wants the value directly:

```python
    @pytest.mark.world_size(3)
    def test_world_size_arg(self, world_size: int) -> None:
        assert world_size == self.world_size
```

The mark only applies to tests inside a `DTest` subclass, and must be applied to the test
method itself: class-level and module-level marks are not currently picked up. Elsewhere the
mark is ignored, so a non-`DTest` test that declares a `world_size` arg will fail with
`fixture 'world_size' not found`.

See `test.py` for more cases, which also serves as the best documentation.
