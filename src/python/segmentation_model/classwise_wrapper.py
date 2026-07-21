# modified to allow usage with MultilassJaccardIndex and the like
# also to use with our custom PanopticQuality metrics
# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import typing

import torch

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE

if typing.TYPE_CHECKING:
    from torch.nn import Module

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ClasswiseWrapper.plot"]

class WrapperMetric(Metric):
    """Abstract base class for wrapper metrics.

    Wrapper metrics are characterized by them wrapping another metric, and forwarding all calls to the wrapped metric.
    This means that all logic regarding synchronization etc. is handled by the wrapped metric, and the wrapper metric
    should not do anything in this regard.

    This class therefore overwrites all methods that are related to synchronization, and does nothing in them.

    Additionally, the forward method is not implemented by default as custom logic is required for each wrapper metric.

    """

    def _wrap_update(self, update: typing.Callable) -> typing.Callable:
        """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
        return update

    def _wrap_compute(self, compute: typing.Callable) -> typing.Callable:
        """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
        return compute

    def forward(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
        raise NotImplementedError


class ClasswiseWrapper(WrapperMetric):
    """Wrapper metric for altering the output of classification metrics.

    This metric works together with classification metrics that returns multiple values (one value per class) such that
    label information can be automatically included in the output.

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a single
            tensor that is split along the first dimension.
        labels: list of strings indicating the different classes.
        prefix: string that is prepended to the metric names.
        postfix: string that is appended to the metric names.

    Example::
        Basic example where the output of a metric is unwrapped into a dictionary with the class index as keys:

        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_0': tensor(0.5000),
        'multiclassaccuracy_1': tensor(0.7500),
        'multiclassaccuracy_2': tensor(0.)}

    Example::
        Using custom name via prefix and postfix:

        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> metric_pre = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), prefix="acc-")
        >>> metric_post = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), postfix="-acc")
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric_pre(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'acc-0': tensor(0.5000),
         'acc-1': tensor(0.7500),
         'acc-2': tensor(0.)}
        >>> metric_post(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'0-acc': tensor(0.5000),
         '1-acc': tensor(0.7500),
         '2-acc': tensor(0.)}

    Example::
        Providing labels as a list of strings:

        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> metric = ClasswiseWrapper(
        ...    MulticlassAccuracy(num_classes=3, average=None),
        ...    labels=["horse", "fish", "dog"]
        ... )
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_horse': tensor(0.3333),
        'multiclassaccuracy_fish': tensor(0.6667),
        'multiclassaccuracy_dog': tensor(0.)}

    Example::
        Classwise can also be used in combination with :class:`~torchmetrics.MetricCollection`. In this case, everything
        will be flattened into a single dictionary:

        >>> from torchmetrics import MetricCollection
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall
        >>> labels = ["horse", "fish", "dog"]
        >>> metric = MetricCollection(
        ...     {'multiclassaccuracy': ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), labels),
        ...     'multiclassrecall': ClasswiseWrapper(MulticlassRecall(num_classes=3, average=None), labels)}
        ... )
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_horse': tensor(0.),
         'multiclassaccuracy_fish': tensor(0.3333),
         'multiclassaccuracy_dog': tensor(0.4000),
         'multiclassrecall_horse': tensor(0.),
         'multiclassrecall_fish': tensor(0.3333),
         'multiclassrecall_dog': tensor(0.4000)}

    """

    def __init__(self,
        metric: Metric,
        labels: typing.Optional[typing.List[str]] = None,
        prefix: typing.Optional[str] = None,
        postfix: typing.Optional[str] = None) -> None:
        super().__init__()

        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")
        self.metric = metric

        if labels is not None and not (isinstance(labels, list) and all(isinstance(lab, str) for lab in labels)):
            raise ValueError(f"Expected argument `labels` to either be `None` or a list of strings but got {labels}")
        self.labels = labels

        if prefix is not None and not isinstance(prefix, str):
            raise ValueError(f"Expected argument `prefix` to either be `None` or a string but got {prefix}")
        self._prefix = prefix

        if postfix is not None and not isinstance(postfix, str):
            raise ValueError(f"Expected argument `postfix` to either be `None` or a string but got {postfix}")
        self._postfix = postfix

        self._update_count = 1

    def _convert(self, x: torch.Tensor) -> typing.Dict[str, typing.Any]:
        if x.shape[0] == 1:
            x = x[0]
        # Will set the class name as prefix if neither prefix nor postfix is given
        if not self._prefix and not self._postfix:
            prefix = f"{self.metric.__class__.__name__.lower()}_"
            postfix = ""
        else:
            prefix = self._prefix or ""
            postfix = self._postfix or ""
        if self.labels is None:
            return {f"{prefix}{i}{postfix}": val for i, val in enumerate(x)}
        return {f"{prefix}{lab}{postfix}": val for lab, val in zip(self.labels, x)}

    def forward(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Calculate on batch and accumulate to global state."""
        return self._convert(self.metric(*args, **kwargs))

    def update(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Update state."""
        self.metric.update(*args, **kwargs)

    def compute(self) -> typing.Dict[str, torch.Tensor]:
        """Compute metric."""
        return self._convert(self.metric.compute())

    def reset(self) -> None:
        """Reset metric."""
        self.metric.reset()

    def __getattr__(self, name: str) -> typing.Union[torch.Tensor, "Module"]:
        """Get attribute from classwise wrapper."""
        if name == "metric" or (name in self.__dict__ and name not in self.metric.__dict__):
            # we need this to prevent from infinite getattribute loop.
            return super().__getattr__(name)

        return getattr(self.metric, name)

    def __setattr__(self, name: str, value: typing.Any) -> None:
        """Set attribute to classwise wrapper."""
        if hasattr(self, "metric") and name in self.metric._defaults:
            setattr(self.metric, name, value)
        else:
            super().__setattr__(name, value)
            if name == "metric":
                self._defaults = self.metric._defaults
                self._persistent = self.metric._persistent
                self._reductions = self.metric._reductions