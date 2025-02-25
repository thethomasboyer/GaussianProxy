# Installation with [`uv`](https://docs.astral.sh/uv/)

1. Start installing the environment with:

```sh
uv sync --frozen
```
2. Finish installing the environment with:

```sh
git clone https://github.com/thethomasboyer/torch-fidelity-symlinks.git src/torch-fidelity
uv pip install --editable src/torch-fidelity
```

This will install a [cutom fork](https://github.com/thethomasboyer/torch-fidelity-symlinks) of [`torch-fidelity`](https://github.com/toshas/torch-fidelity) with very minimal modifications.
