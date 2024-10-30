"""
deps_listing_diffs.py

This script compares the package dependencies listed in `requirements.txt` and `environment.yaml`.

It performs the following steps:
1. Reads and normalizes package names from `requirements.txt`.
2. Reads and normalizes package names from `environment.yaml`.
3. Identifies and prints duplicate package names in each file.
4. Finds and prints the differences between the two sets of packages, indicating which packages are unique to each file.

Normalization involves converting package names to lowercase and replacing hyphens with underscores to ensure consistency.
"""

from collections import Counter

import yaml


def normalize_package_name(name):
    return name.replace("-", "_").lower()


# Read requirements.txt
with open("requirements.txt", "r") as req_file:
    raw_req_packages = set(
        line.strip().split("==")[0] for line in req_file if line.strip() and not line.startswith("#")
    )
    print(f"Found {len(raw_req_packages)} packages in requirements.txt,", end=" ")
    req_packages = set(normalize_package_name(pkg) for pkg in raw_req_packages)
print(f"{len(req_packages)} left after normalization.", end="")
if len(raw_req_packages) != len(req_packages):
    req_counter = Counter([normalize_package_name(pkg) for pkg in raw_req_packages])
    print("Duplicates:", [pkg for pkg, count in req_counter.items() if count > 1])
else:
    print("")

# Read environment.yaml
with open("environment.yaml", "r") as env_file:
    env_data = yaml.safe_load(env_file)
    raw_env_packages = set(dep.split("=")[0] for dep in env_data["dependencies"] if isinstance(dep, str))
    print(f"Found {len(raw_env_packages)} packages in environment.yaml,", end=" ")
    env_packages = set(normalize_package_name(pkg) for pkg in raw_env_packages)
print(f"{len(env_packages)} packages left after normalization.", end="")
if len(raw_env_packages) != len(env_packages):
    env_counter = Counter([normalize_package_name(pkg) for pkg in raw_env_packages])
    print(" Duplicates:", [pkg for pkg, count in env_counter.items() if count > 1])
else:
    print("")

# Find differences
only_in_requirements = sorted(req_packages - env_packages)
only_in_environment = sorted(env_packages - req_packages)

print(f"\n{len(only_in_requirements)} packages only in requirements.txt: {only_in_requirements}")
print(f"\n{len(only_in_environment)} packages only in environment.yaml: {only_in_environment}")
