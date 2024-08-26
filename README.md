# Installation

1. Install the environment with:

    ```sh
    mamba create -f environment.yaml
    ```

2. Then from inside that environment:

    ```sh
    pip install -e .
    ```

    which will install some other packages not available in Anaconda with `pip`, and the repo itself as a package (in editable mode!).

You can *also* skip the `mamba` step and only do `pip install -e .` inside an virtual environment. In that case make sure you are using Python >= 3.10.

A custom listing of the project's *explicit and direct dependencies* (*à la* `pyproject.toml`) are in `requirements.yaml`: you can also replace the first step with `mamba create -f requirements.yaml` and let `mamba` resolve the declared dependencies to their latest versions. Of courses things might break then...
