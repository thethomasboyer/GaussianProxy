from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

DEVICE = "cuda:3"


def main():
    t = torch.rand(10, 10).to(DEVICE)
    for _ in range(100):
        t = t @ t


def custom_trace_handler(dir_name: str):
    def handler_fn(prof: profile):
        file_name = f"{prof.step_num}.pt.trace.json.gz"
        path = Path(dir_name) / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(path))

    return handler_fn


trace_handler = custom_trace_handler("pytorch_traces")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    on_trace_ready=trace_handler,
) as profiler:
    main()
