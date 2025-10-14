# Distributed Pytest Utils

Shamelessly stolen from/inspired by DeepSpeed, then simplified.

```
uv pip install git+https://github.com/garrett361/dtest
```

Minimal example:

```python

import pytest
class TestDTest(DTest):
    requires_cuda_env = False

    @pytest.mark.parametrize("n", list(range(1, 4)))
    def test_all_reduce(self, n: int) -> None:
        t = torch.arange(n * self.world_size, device=self.device)
        dist.all_reduce(t)
        self.print_rank(f"{t=}")
```
