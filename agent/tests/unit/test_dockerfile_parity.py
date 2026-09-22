"""The two Dockerfiles build the same image.

`agent/Dockerfile` is what `lk agent deploy` and the CI docker-build job use.
`infra/container/Dockerfile` is what Cloudflare Containers builds, from the
repository root instead of from `agent/`. They have to produce the same runtime,
or one deploy target runs an image nobody has tested.

They carried a comment saying "Keep the install/prewarm steps aligned with
agent/Dockerfile", which is a request, not a mechanism. This is the mechanism.

Only the properties that would actually change behaviour are compared -- base
image, Python version, the uid, whether build tools reach the runtime stage.
The paths differ on purpose (`pyproject.toml` versus `agent/pyproject.toml`) and
so does the entrypoint, so those are not compared.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent.parent
AGENT_DOCKERFILE = ROOT / "agent" / "Dockerfile"
CONTAINER_DOCKERFILE = ROOT / "infra" / "container" / "Dockerfile"

DOCKERFILES = (AGENT_DOCKERFILE, CONTAINER_DOCKERFILE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _directive(text: str, name: str) -> list[str]:
    return re.findall(rf"^{name}\s+(.*)$", text, re.M)


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.parent.name)
def test_each_dockerfile_exists(path: Path) -> None:
    assert path.exists(), f"{path} is missing"


def test_both_pin_the_same_python_version() -> None:
    versions = {_directive(_read(p), "ARG")[0] for p in DOCKERFILES}

    assert len(versions) == 1, f"Python versions disagree: {versions}"


def test_both_use_the_same_base_image() -> None:
    bases = {frozenset(_directive(_read(p), "FROM")) for p in DOCKERFILES}

    assert len(bases) == 1, "the two images are built on different bases"


def test_both_run_as_the_same_non_root_user() -> None:
    for path in DOCKERFILES:
        text = _read(path)
        assert '--uid "${UID}"' in text
        assert _directive(text, "USER") == ["appuser"], f"{path} does not drop privileges"

    uids = {re.search(r"ARG UID=(\d+)", _read(p)).group(1) for p in DOCKERFILES}
    assert uids == {"10001"}, f"uids disagree: {uids}"


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.parent.name)
def test_the_build_toolchain_does_not_reach_the_runtime_stage(path: Path) -> None:
    """gcc, g++ and python3-dev build wheels and are needed by nothing at run
    time. Shipping them leaves a compiler inside a container that runs
    model-chosen code paths and holds a dozen provider credentials."""
    text = _read(path)
    builder, _, runtime = text.partition(
        "FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS runtime"
    )

    assert "gcc" in builder, "the builder stage should install the toolchain"
    assert "gcc" not in runtime, f"{path} ships gcc in the runtime image"
    assert "python3-dev" not in runtime, f"{path} ships python3-dev in the runtime image"


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.parent.name)
def test_dependencies_are_installed_from_the_lockfile(path: Path) -> None:
    """`--locked` is what makes the image reproducible; without it a rebuild can
    resolve different versions than the ones the suite ran against."""
    assert "uv sync --locked" in _read(path)


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.parent.name)
def test_the_vad_model_is_prewarmed_at_build_time(path: Path) -> None:
    """Otherwise the first session of a cold container waits on a download."""
    assert "download-files" in _read(path)


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.parent.name)
def test_the_runtime_stage_copies_the_built_environment(path: Path) -> None:
    assert "COPY --from=builder" in _read(path)


def test_both_reference_each_other() -> None:
    """A reader who opens one should learn the other exists."""
    assert "infra/container/Dockerfile" in _read(AGENT_DOCKERFILE)
    assert "agent/Dockerfile" in _read(CONTAINER_DOCKERFILE)
