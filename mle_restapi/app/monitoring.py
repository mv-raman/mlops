from prometheus_fastapi_instrumentator import Instrumentator, metrics

from typing import Callable
from prometheus_fastapi_instrumentator.metrics import Info
from prometheus_client import Counter, Histogram, Summary
import os
import numpy as np
import json


NAMESPACE = os.environ.get("METRICS_NAMESPACE", "mle_fastapi")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "classify")


instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/docs", "/health"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    body_handlers=[r".*"],
    inprogress_labels=True,
)


def http_requested_classes() -> Callable[[Info], None]:
    METRIC = Counter(
        "http_requested_classes",
        "Number of times a certain class has been requested.",
        labelnames=("classes",)
    )

    def instrumentation(info: Info) -> None:
        try:
            classes = set()
            classes_str = json.loads(info.response.body)['label']
            classes.add(classes_str)
            METRIC.labels(classes).inc()
        except Exception as e:
            pass

    return instrumentation


# golden gate metrics
def c4xx_total(
    metric_name: str = "c4XX_total",
    metric_doc: str = "c4XX_total Number of the service spans results in 4xx error",
    metric_namespace: str = "",
    metric_subsystem: str = "",
) -> Callable[[Info], None]:
    METRIC = Counter(
        name= metric_name,
        documentation= metric_doc,
        namespace= metric_namespace,
        subsystem=metric_subsystem
        )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/classify":
            if 400 <= info.response.status_code <500 :
                METRIC.inc()

    return instrumentation


def c2xx_total(
    metric_name: str = "c2XX_total",
    metric_doc: str = "c2XX_total count Number of the service spans results in 2** error",
    metric_namespace: str = "",
    metric_subsystem: str = "",
) -> Callable[[Info], None]:
    METRIC = Counter(
        name= metric_name,
        documentation= metric_doc,
        namespace= metric_namespace,
        subsystem=metric_subsystem
        )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/classify":
            if info.response.status_code ==200 :
                METRIC.inc()

    return instrumentation


# ----- add metrics -----
instrumentator.add(
    metrics.request_size(
        metric_name="request_size_bytes",
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.response_size(
        metric_name="response_size_bytes",
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.latency(
        metric_name="latency_seconds",
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.requests(
        metric_name="requests_number",
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(http_requested_classes())

instrumentator.add(
    c4xx_total(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM)
)
instrumentator.add(
    c2xx_total(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM)
)