# cuda-kernel-verifier

**Runtime correctness checker for custom CUDA / Triton kernels - ~200 lines of logic.**

Attach a single decorator to any forward function and the library will periodically re-run the same inputs through a known-correct implementation in a background thread, comparing results with `torch.allclose`. Zero impact on the training graph. Works with raw kernels, Triton ops, `torch.autograd.Function`, or any `nn.Module`, including models and layers compiled with `torch.compile`. The enqueue call is decorated with `@torch.compiler.disable` so it is always a clean graph break with no interference with compiled regions.

---

## How it works

```
forward(x) ──► kernel result ──► returned to caller immediately
                    │
                    ▼  (background thread, non-blocking)
             outlier check
                    │
             ┌──────┴──────┐
             │ outlier?    │ not outlier?
             │             │
             ▼             ▼
          enqueue     random sample gate
                      (execution_sample_probability)
                             │
                             ▼
                      ground_truth(x)
                             │
                             ▼
                      torch.allclose?
                        │         │
                       yes        no
                        │         │
                      discard   failure_callback(...)
```

### Sampling

The checker does **not** run the ground truth on every call. That would negate the point of writing a fast kernel. Instead, each call passes through two gates before work is enqueued:

1. **Outlier gate** - if the current input is detected as an outlier (see below), it is enqueued unconditionally, so unusual inputs are never skipped.
2. **Random gate** - otherwise, the call is enqueued with probability `execution_sample_probability` (default `0.5`). Tune this down for large models where verification overhead matters.

The comparison itself runs in a single daemon background thread so the main training loop is never blocked. You can adjust the sampling rate at any point during a run with `EquivalenceChecker.set_execution_sample_probability(p)`, or stop verification entirely with `EquivalenceChecker.stop()`.

### Outlier detection

`ExponentialRunningCentroidExecutionOutlierDetector` tracks the distribution of activations seen so far and flags batches that look statistically different from the norm.

**Algorithm:**

1. Maintain a **running centroid** via exponential moving average:
   `centroid ← α · mean(batch) + (1 − α) · centroid`
   Default `α = 0.01` (slow drift, stable reference).

2. Compute the **L2 distance** of each sample in the batch from the centroid.

3. Append distances to a rolling window of up to `max_distances` values (default 10 000).

4. A batch is an **outlier** when:
   `mean(distances) / quantile(all_distances, p) ≥ outlier_threshold`
   Default `p = 0.95`, `outlier_threshold = 0.8`.

5. The **first batch is always treated as an outlier** so the centroid can be seeded before any comparison.

This means the verifier is biased toward checking inputs that are unusual (the cases most likely to expose a kernel bug) while randomly sampling the rest.

---

## Installation

**Requires CUDA** Install PyTorch for CUDA first, then the package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install cuda-kernel-verifier
```

---

## Quick start

```python
import torch
from cuda_kernel_verifier import equivalent, EquivalenceChecker

def ground_truth(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=1)

def on_mismatch(args: FailureCallbackArgs) -> None:
    diff = (args.original_result - args.ground_truth_result).abs().max().item()
    raise AssertionError(f"Kernel diverged! max abs diff = {diff:.6f}")

@equivalent(ground_truth, on_mismatch, rtol=1e-1, atol=1e-6)
def my_fast_row_sum(x: torch.Tensor) -> torch.Tensor:
    return my_cuda_row_sum_kernel(x)

EquivalenceChecker.start(execution_sample_probability=0.5)

result = my_fast_row_sum(torch.randn(128, 512, device="cuda"))

EquivalenceChecker.stop()
```

### Attaching to `torch.autograd.Function`

```python
from torch.autograd import Function
from cuda_kernel_verifier import equivalent, FailureCallbackArgs

def sum_ground_truth(ctx, x):
    return x.sum(dim=1)

def on_mismatch(args: FailureCallbackArgs) -> None:
    raise AssertionError("kernel diverged!")

class FastRowSum(Function):
    @staticmethod
    @equivalent(sum_ground_truth, on_mismatch, rtol=1e-1, atol=1e-6)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return my_cuda_kernel(x)
```

The decorator wraps the static method, so `ctx` is passed through transparently. Just mirror the full signature in the ground truth and ignore `ctx` with `_` if needed.

### Custom outlier detector

```python
from cuda_kernel_verifier import (
    equivalent,
    ExponentialRunningCentroidExecutionOutlierDetector,
)

detector = ExponentialRunningCentroidExecutionOutlierDetector(
    percentile=0.99,
    outlier_threshold=0.9,
    exponential_alpha=5e-3,
)

@equivalent(ground_truth, outlier_detector=detector)
def my_kernel(x):
    ...
```

---

## API reference

### `equivalent(ground_truth_function, failure_callback=None, *, rtol=1e-2, atol=1e-8, outlier_detector=None)`

Decorator factory. Returns a decorator that wraps the target function.

| Parameter               | Description                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------- |
| `ground_truth_function` | Known-correct implementation with the same signature.                               |
| `failure_callback`      | Called with `FailureCallbackArgs` on mismatch. Required.                            |
| `rtol`                  | Relative tolerance for `torch.allclose` (default `1e-2`).                           |
| `atol`                  | Absolute tolerance for `torch.allclose` (default `1e-8`).                           |
| `outlier_detector`      | Outlier strategy. Defaults to `ExponentialRunningCentroidExecutionOutlierDetector`. |

---

### `EquivalenceChecker`

Class-level singleton that manages the background thread and queue.

| Method                                                                | Description                                                |
| --------------------------------------------------------------------- | ---------------------------------------------------------- |
| `start(max_execution_queue_size=0, execution_sample_probability=0.5)` | Start the background thread. Resets all outlier detectors. |
| `stop()`                                                              | Stop the thread and drain the queue.                       |
| `is_running()`                                                        | Returns `True` if the checker is active.                   |
| `set_execution_sample_probability(p)`                                 | Adjust sampling rate at runtime.                           |

---

### `ExponentialRunningCentroidExecutionOutlierDetector`

| Parameter           | Default  | Description                                                            |
| ------------------- | -------- | ---------------------------------------------------------------------- |
| `percentile`        | `0.95`   | Quantile used as the distance scale reference.                         |
| `max_distances`     | `10_000` | Rolling window size for historical distances.                          |
| `exponential_alpha` | `1e-2`   | EMA factor for the running centroid.                                   |
| `outlier_threshold` | `0.8`    | Fraction of the percentile scale that triggers outlier classification. |

---

### `FailureCallbackArgs`

Dataclass passed to the failure callback.

| Field                 | Type           | Description                                 |
| --------------------- | -------------- | ------------------------------------------- |
| `original_result`     | `torch.Tensor` | Output of the kernel under test (detached). |
| `ground_truth_result` | `torch.Tensor` | Output of the reference function.           |

---

## Full example

See [`examples/mnist_triton.py`](examples/mnist_triton.py) for a complete MNIST training loop using a Triton row-sum kernel validated in real time.

---
