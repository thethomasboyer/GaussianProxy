import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor

from tqdm.rich import tqdm

NB_REPEATS = 100


def measure_import_time(import_statement: str) -> tuple[float, float, float]:
    times = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                subprocess.run,
                f"python3 -c 'import timeit; print(timeit.timeit(\"{import_statement}\", number=1))'",
                shell=True,
                capture_output=True,
                text=True,
            )
            for _ in range(NB_REPEATS)
        ]
        for future in tqdm(futures):
            result = future.result()
            times.append(float(result.stdout.strip()))
    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times)
    median_time = statistics.median(times)
    return mean_time, std_dev, median_time


torch_import_time, torch_std_dev, torch_median = measure_import_time("import torch; t = torch.tensor([1, 2, 3])")
selective_import_time, selective_std_dev, selective_median = measure_import_time(
    "from torch import Tensor, randn, float32, float16, bfloat16, set_grad_enabled, set_float32_matmul_precision, compile, stack, rand, sort, randn_like, tensor, full, maximum, minimum, int64, where, save, flip, rot90, linspace, cat, from_numpy; t = tensor([1, 2, 3])"
)

print(
    "'import torch': mean =",
    round(torch_import_time, 4),
    "std dev =",
    round(torch_std_dev, 4),
    "median =",
    round(torch_median, 4),
)
print(
    "'from torch import ...': mean =",
    round(selective_import_time, 4),
    "std dev =",
    round(selective_std_dev, 4),
    "median =",
    round(selective_median, 4),
)
