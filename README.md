# Installation

1. Start installing the environment with:

    ```sh
    uv sync --frozen
    ```
2. Finish installing the environment with:

```sh
pip install --user -e git+https://github.com/thethomasboyer/torch-fidelity-symlinks.git@master#egg=torch-fidelity --config-settings editable_mode=strict
```

This will install a [cutom fork](https://github.com/thethomasboyer/torch-fidelity-symlinks) of [`torch-fidelity`](https://github.com/toshas/torch-fidelity) with very minimal modifications.
