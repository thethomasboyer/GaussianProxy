# Installation

1. Install the environment with:

    ```sh
    mamba create -f environment.yaml
    ```

2. Then from inside that environment:

    ```sh
    pip install opencv-python-headless
    pip install -e git+https://github.com/thethomasboyer/torch-fidelity-symlinks.git@master#egg=torch-fidelity --config-settings editable_mode=strict
    ```

_Alternatively_, a custom listing of the project's *explicit and direct dependencies* (*à la* `pyproject.toml`) are in `requirements.yaml`: you can also replace the first step with `mamba create -f requirements.yaml` and let `mamba` resolve the declared dependencies (possibly to their latest versions). Of courses things might break then...
