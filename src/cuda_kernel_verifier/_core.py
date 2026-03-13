import torch
import random
import logging
from typing import Callable, Any, Optional
from queue import Queue, Empty, Full
from threading import Thread, Lock
from functools import wraps
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ExecutionOutlierDetector(ABC):
    @abstractmethod
    def is_outlier(self, data: torch.Tensor) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

class ExponentialRunningCentroidExecutionOutlierDetector(ExecutionOutlierDetector):
    """
    Detects outlier inputs by tracking a running centroid and comparing each
    batch's distance from it against a rolling percentile threshold.

    An input is considered an outlier when its mean distance from the centroid
    exceeds ``outlier_threshold * percentile(reference_distances, percentile)``.
    The first batch is always treated as an outlier so the centroid can be
    bootstrapped before comparisons begin.

    Args:
        percentile: Quantile of historical distances used as the scale reference (default 0.95).
        max_distances: Maximum number of past distance values to retain (default 10_000).
        exponential_alpha: EMA smoothing factor for the running centroid (default 1e-2).
        outlier_threshold: Fraction of the percentile scale above which a batch is flagged (default 0.8).
    """

    def __init__(
        self,
        percentile: float = 0.95,
        max_distances: int = 10_000,
        exponential_alpha: float = 1e-2,
        outlier_threshold: float = 0.8,
    ):
        self._percentile = percentile
        self._max_distances = max_distances
        self._exponential_alpha = exponential_alpha
        self._outlier_threshold = outlier_threshold
        self._lock = Lock()
        self.reset()

    def is_outlier(self, data: torch.Tensor) -> bool:
        with self._lock:
            data = data.reshape(len(data), -1)
            batch_mean = data.mean(dim=0)
            if self._centroid is None:
                self._centroid = batch_mean
            else:
                self._centroid = (
                    self._exponential_alpha * batch_mean
                    + (1 - self._exponential_alpha) * self._centroid
                )

            distance = torch.norm(data - self._centroid, dim=1)
            self._reference_distances = torch.cat(
                [self._reference_distances.to(distance.device), distance]
            )[-self._max_distances :]

            if self._first_batch:
                self._first_batch = False
                return True

            quantile_scale = torch.quantile(self._reference_distances, self._percentile)
            return (distance.mean() / quantile_scale).clamp(0, 1).item() >= self._outlier_threshold

    def reset(self) -> None:
        with self._lock:
            self._centroid: Optional[torch.Tensor] = None
            self._reference_distances: torch.Tensor = torch.empty(0)
            self._first_batch = True

@dataclass
class FailureCallbackArgs:
    """Passed to the failure callback when a mismatch is detected."""
    original_result: torch.Tensor
    ground_truth_result: torch.Tensor

@dataclass
class OperationData:
    identifier: str
    outlier_detector: ExecutionOutlierDetector

@dataclass
class EquivalenceCheckerExecutionArgs:
    original_result: torch.Tensor
    ground_truth_function: Callable[..., torch.Tensor]
    failure_callback: Callable[[FailureCallbackArgs], None]
    rtol: float
    atol: float
    outlier_detector: ExecutionOutlierDetector
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

class EquivalenceChecker:
    """
    Singleton background checker that asynchronously validates kernel outputs
    against a ground-truth function.

    Call :meth:`start` once at the beginning of a run and :meth:`stop` when done.
    The checker runs a single background thread that dequeues samples and
    compares them via :func:`torch.allclose`.
    """

    _EXECUTION_GET_READ_TIMEOUT = 1

    _running = False
    _running_lock = Lock()
    _execution_sample_probability = 0.5
    _execution_sample_probability_lock = Lock()
    _outlier_detectors: dict[str, ExecutionOutlierDetector] = {}
    _outlier_detectors_lock = Lock()
    _execution_queue: Queue[EquivalenceCheckerExecutionArgs] = Queue()

    @classmethod
    def start(cls, max_execution_queue_size: int = 0, execution_sample_probability: float = 0.5) -> None:
        """
        Start the background equivalence-checking thread.

        Args:
            max_execution_queue_size: Maximum pending checks before drops occur. 0 = unlimited.
            execution_sample_probability: Fraction of non-outlier calls to sample (0.0–1.0).
        """
        with cls._running_lock:
            if cls._running:
                return
            cls._running = True

        with cls._execution_sample_probability_lock:
            cls._execution_sample_probability = execution_sample_probability

        with cls._outlier_detectors_lock:
            for outlier_detector in cls._outlier_detectors.values():
                outlier_detector.reset()

        cls._execution_queue = Queue(max_execution_queue_size)

        execution_thread_handler = Thread(target=cls._execution_thread, daemon=True)
        execution_thread_handler.start()

    @classmethod
    def stop(cls) -> None:
        """Stop the background thread and drain the queue."""
        with cls._running_lock:
            if not cls._running:
                return
            cls._running = False

        try:
            while True:
                cls._execution_queue.get_nowait()
        except Empty:
            pass

        with cls._outlier_detectors_lock:
            for outlier_detector in cls._outlier_detectors.values():
                outlier_detector.reset()

    @classmethod
    def is_running(cls) -> bool:
        with cls._running_lock:
            return cls._running

    @classmethod
    def set_execution_sample_probability(cls, execution_sample_probability: float) -> None:
        with cls._execution_sample_probability_lock:
            cls._execution_sample_probability = execution_sample_probability

    @classmethod
    def register_operation(cls, operation_data: OperationData) -> None:
        with cls._outlier_detectors_lock:
            cls._outlier_detectors[operation_data.identifier] = operation_data.outlier_detector

    @classmethod
    @torch.compiler.disable
    def enqueue_equivalence_check(cls, args: EquivalenceCheckerExecutionArgs) -> None:
        if not cls.is_running():
            return
        if not args.outlier_detector.is_outlier(data=args.original_result):
            with cls._execution_sample_probability_lock:
                if random.random() >= cls._execution_sample_probability:
                    return

        try:
            cls._execution_queue.put_nowait(args)
        except Full:
            logger.warning("Execution queue is full - dropping equivalence check")
        except Exception:
            logger.error("Error enqueuing equivalence check", exc_info=True)

    @classmethod
    def _execution_thread(cls) -> None:
        while cls.is_running():
            try:
                try:
                    execution_data = cls._execution_queue.get(
                        block=True, timeout=cls._EXECUTION_GET_READ_TIMEOUT
                    )
                    ground_truth_result = execution_data.ground_truth_function(
                        *execution_data.args, **execution_data.kwargs
                    )
                    if not torch.allclose(
                        execution_data.original_result,
                        ground_truth_result,
                        rtol=execution_data.rtol,
                        atol=execution_data.atol,
                    ):
                        execution_data.failure_callback(
                            FailureCallbackArgs(
                                original_result=execution_data.original_result,
                                ground_truth_result=ground_truth_result,
                            )
                        )
                except Empty:
                    continue
            except Exception:
                logger.error("Error executing equivalence check", exc_info=True)

def equivalent(
    ground_truth_function: Callable,
    failure_callback: Callable[[FailureCallbackArgs], None],
    *,
    rtol: float = 1e-2,
    atol: float = 1e-8,
    outlier_detector: Optional[ExecutionOutlierDetector] = None,
) -> Callable:
    """
    Decorator that asynchronously validates a function's output against a
    ground-truth implementation.

    Attach this to any callable - a plain function, a ``torch.autograd.Function``
    static method, or an ``nn.Module.forward`` - and it will periodically
    re-run the same inputs through ``ground_truth_function`` in a background
    thread and compare the results with :func:`torch.allclose`.

    The decorator does **not** affect the return value or the training graph;
    the original result is returned immediately and detached before being sent
    to the background checker.

    Args:
        ground_truth_function: A known-correct implementation with the same
            signature as the decorated function.
        failure_callback: Called when a mismatch is detected. Receives a
            :class:`FailureCallbackArgs`. Required - you must decide what
            happens on failure (raise, log, alert, etc.).
        rtol: Relative tolerance forwarded to :func:`torch.allclose` (default 1e-2).
        atol: Absolute tolerance forwarded to :func:`torch.allclose` (default 1e-8).
        outlier_detector: Strategy used to decide whether an input is an outlier
            and should bypass the random sampling gate. Defaults to
            :class:`ExponentialRunningCentroidExecutionOutlierDetector`.

    Example::

        def my_ground_truth(x):
            return x.sum(dim=1)

        def on_mismatch(args):
            raise AssertionError("kernel diverged!")

        @equivalent(my_ground_truth, on_mismatch, rtol=1e-1, atol=1e-6)
        def my_cuda_forward(x):
            return cuda_row_sum_kernel(x)
    """
    if outlier_detector is None:
        outlier_detector = ExponentialRunningCentroidExecutionOutlierDetector()

    def match_decorator(function: Callable[..., torch.Tensor]) -> Callable:
        EquivalenceChecker.register_operation(
            operation_data=OperationData(
                identifier=function.__name__,
                outlier_detector=outlier_detector,
            )
        )

        @wraps(function)
        def wrapped_function(*args: Any, **kwargs: Any) -> Any:
            original_result = function(*args, **kwargs)
            EquivalenceChecker.enqueue_equivalence_check(
                args=EquivalenceCheckerExecutionArgs(
                    original_result=original_result.detach().requires_grad_(False),
                    ground_truth_function=ground_truth_function,
                    failure_callback=failure_callback,
                    rtol=rtol,
                    atol=atol,
                    outlier_detector=outlier_detector,
                    args=args,
                    kwargs=kwargs,
                )
            )
            return original_result

        return wrapped_function

    return match_decorator
