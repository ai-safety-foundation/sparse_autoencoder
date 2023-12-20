"""Find metric result."""
from sparse_autoencoder.metrics.abstract_metric import MetricLocation, MetricResult


def find_metric_result(
    metrics: list[MetricResult],
    *,
    location: MetricLocation | None = None,
    name: str | None = None,
    postfix: str | None = None,
) -> MetricResult:
    """Find exactly one metric result from a list of results.

    Motivation:
        For automated testing, it's useful to search for a specific result and check it is as
        expected.

    Example:
        >>> import torch
        >>> metric_results = [
        ...     MetricResult(
        ...         component_wise_values=torch.tensor([1.0, 2.0, 3.0]),
        ...         location=MetricLocation.TRAIN,
        ...         name="loss",
        ...         postfix="baseline_loss",
        ...     ),
        ...     MetricResult(
        ...         component_wise_values=torch.tensor([4.0, 5.0, 6.0]),
        ...         location=MetricLocation.TRAIN,
        ...         name="loss",
        ...         postfix="loss_with_reconstruction",
        ...     )
        ... ]
        >>> find_metric_result(
        ...     metric_results, name="loss", postfix="baseline_loss"
        ... ).component_wise_values
        tensor([1., 2., 3.])

    Args:
        metrics: List of metric results.
        location: Location of the metric to find. None means all locations.
        name: Name of the metric to find. None means all names.
        postfix: Postfix of the metric to find. None means **no postfix**.

    Returns:
        Metric result.

    Raises:
        ValueError: If the metric is not found.
    """
    if name is None and postfix is None and location is None:
        error_message = "At least one of name, postfix or location must be provided."
        raise ValueError(error_message)

    results: list[MetricResult] = []

    for metric in metrics:
        if (
            (metric.location == location or location is None)
            and (metric.name == name or name is None)
            and (metric.postfix == postfix)
        ):
            results.append(metric)  # noqa: PERF401

    if len(results) == 0:
        metric_names = ",\n ".join([f"{metric.name} {metric.postfix or ''}" for metric in metrics])
        error_message = f"Metric not found. The only metrics found were:\n {metric_names}"
        raise ValueError(error_message)

    if len(results) == 1:
        return results[0]

    error_message = f"Multiple metrics found: name={name}, postfix={postfix}, location={location}"
    raise ValueError(error_message)
