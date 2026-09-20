"""Pytest hooks and fixtures. Markers are declared in pyproject.toml."""

import pytest

import jax

jax.config.update("jax_enable_x64", True)


def pytest_addoption(parser):
    parser.addoption(
        "--samples",
        type=int,
        default=3,
        help="random matrices drawn per size (default 3; the full run uses 20)",
    )


@pytest.fixture(scope="session")
def samples(request):
    return request.config.getoption("--samples")


def _cuda_available():
    try:
        return bool(jax.devices("cuda"))
    except RuntimeError:
        return False


CUDA_AVAILABLE = _cuda_available()


def pytest_collection_modifyitems(config, items):
    if CUDA_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
