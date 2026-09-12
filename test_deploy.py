"""Unit tests for deploy/benchmark-ui.container Quadlet specification.

Verifies persistence volume mounts, loopback network bindings, SELinux labels,
and required systemd/Quadlet sections.
"""

from collections import defaultdict
from pathlib import Path

import pytest

DEPLOY_CONTAINER_PATH = Path(__file__).parent / "deploy" / "benchmark-ui.container"


def parse_quadlet_file(path: Path) -> dict[str, dict[str, list[str]]]:
    """Parse a systemd Quadlet container file into sections and key-value pairs.

    Returns a dict mapping section name to a dict of key -> list of string values.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Quadlet file not found: {path}")

    sections: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    current_section: str | None = None

    for line_num, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue

        if current_section is None:
            raise ValueError(f"Line {line_num} outside of any section: {line}")

        if "=" not in line:
            raise ValueError(f"Malformed key-value line at {line_num}: {line}")

        key, val = line.split("=", 1)
        sections[current_section][key.strip()].append(val.strip())

    return {sec: dict(kvs) for sec, kvs in sections.items()}


def validate_quadlet_config(config: dict[str, dict[str, list[str]]]) -> None:
    """Validate mandatory deployment security and persistence invariants."""
    required_sections = {"Unit", "Container", "Service", "Install"}
    missing = required_sections - set(config.keys())
    if missing:
        raise ValueError(f"Missing required sections: {sorted(missing)}")

    container = config["Container"]

    # Invariant 1: Volume persistence for benchmark history with SELinux :Z
    expected_volume = "/mnt/DATA/boy/prod/benchmark-ui/data/history:/app/history:Z"
    volumes = container.get("Volume", [])
    if expected_volume not in volumes:
        raise ValueError(
            f"Missing required persistent history volume: {expected_volume} (found: {volumes})"
        )

    # Invariant 2: Loopback binding only (127.0.0.1), no public/host exposure
    ports = container.get("PublishPort", [])
    if not ports:
        raise ValueError("Missing PublishPort configuration in Container section")
    for port in ports:
        if not port.startswith("127.0.0.1:"):
            raise ValueError(
                f"PublishPort must bind to 127.0.0.1 loopback only: {port}"
            )

    # Invariant 3: ContainerName and Image specification
    if not container.get("ContainerName"):
        raise ValueError("Missing ContainerName in Container section")
    if not container.get("Image"):
        raise ValueError("Missing Image in Container section")

    # Invariant 4: WUD digest label for registry update tracking
    labels = container.get("Label", [])
    if "wud.watch.digest=true" not in labels:
        raise ValueError("Missing Label=wud.watch.digest=true for update tracking")


def test_quadlet_file_exists():
    """Verify that deploy/benchmark-ui.container exists on disk."""
    assert DEPLOY_CONTAINER_PATH.is_file(), f"{DEPLOY_CONTAINER_PATH} does not exist"


def test_quadlet_valid_sections():
    """Verify all standard Quadlet sections are present."""
    config = parse_quadlet_file(DEPLOY_CONTAINER_PATH)
    assert "Unit" in config
    assert "Container" in config
    assert "Service" in config
    assert "Install" in config


def test_quadlet_persistence_volume():
    """Verify Volume mount points to host data directory with private :Z SELinux label."""
    config = parse_quadlet_file(DEPLOY_CONTAINER_PATH)
    container = config["Container"]
    volumes = container.get("Volume", [])

    expected_mount = "/mnt/DATA/boy/prod/benchmark-ui/data/history:/app/history:Z"
    assert expected_mount in volumes, (
        f"Expected volume '{expected_mount}' not found in Container volumes: {volumes}"
    )

    # Parse and assert volume components explicitly
    matched = False
    for vol in volumes:
        parts = vol.split(":")
        if len(parts) == 3:
            host_path, container_path, selinux_opt = parts
            if host_path == "/mnt/DATA/boy/prod/benchmark-ui/data/history":
                assert container_path == "/app/history"
                assert selinux_opt == "Z"
                matched = True
    assert matched, "Failed to match parsed volume components"


def test_quadlet_network_isolation_loopback():
    """Verify container ports are strictly bound to 127.0.0.1 loopback."""
    config = parse_quadlet_file(DEPLOY_CONTAINER_PATH)
    container = config["Container"]
    ports = container.get("PublishPort", [])

    assert ports == ["127.0.0.1:8501:8501"]
    for port in ports:
        assert port.startswith("127.0.0.1:")
        assert "0.0.0.0" not in port


def test_quadlet_container_metadata():
    """Verify container naming, image, autoupdate, and WUD label."""
    config = parse_quadlet_file(DEPLOY_CONTAINER_PATH)
    container = config["Container"]

    assert container.get("ContainerName") == ["production-benchmark-ui"]
    assert container.get("Image") == ["ghcr.io/sheepdestroyer/benchmark-ui:latest"]
    assert container.get("AutoUpdate") == ["registry"]
    assert "wud.watch.digest=true" in container.get("Label", [])


def test_quadlet_healthcheck():
    """Verify health check probe is configured properly."""
    config = parse_quadlet_file(DEPLOY_CONTAINER_PATH)
    container = config["Container"]

    assert "HealthCmd" in container
    assert "8501/_stcore/health" in container["HealthCmd"][0]
    assert container.get("HealthInterval") == ["15s"]
    assert container.get("HealthTimeout") == ["5s"]
    assert container.get("HealthRetries") == ["5"]


def test_quadlet_environment_variables():
    """Verify Streamlit environment configurations."""
    config = parse_quadlet_file(DEPLOY_CONTAINER_PATH)
    container = config["Container"]
    env_vars = container.get("Environment", [])

    assert "STREAMLIT_SERVER_PORT=8501" in env_vars
    assert "STREAMLIT_SERVER_ADDRESS=0.0.0.0" in env_vars
    assert "STREAMLIT_SERVER_HEADLESS=true" in env_vars


def test_quadlet_service_and_install():
    """Verify systemd Service and Install section settings."""
    config = parse_quadlet_file(DEPLOY_CONTAINER_PATH)

    assert config["Service"].get("Restart") == ["on-failure"]
    assert config["Service"].get("TimeoutStartSec") == ["180"]
    assert config["Install"].get("WantedBy") == ["default.target"]


def test_validate_quadlet_config_success():
    """Verify validation passes on the vendored Quadlet file."""
    config = parse_quadlet_file(DEPLOY_CONTAINER_PATH)
    validate_quadlet_config(config)


def test_parse_quadlet_file_not_found(tmp_path):
    """Verify FileNotFoundError on nonexistent file."""
    nonexistent = tmp_path / "missing.container"
    with pytest.raises(FileNotFoundError, match="Quadlet file not found"):
        parse_quadlet_file(nonexistent)


def test_parse_quadlet_file_outside_section(tmp_path):
    """Verify ValueError when lines occur before any section header."""
    bad_file = tmp_path / "bad.container"
    bad_file.write_text("Key=Value\n[Unit]\nDescription=Test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside of any section"):
        parse_quadlet_file(bad_file)


def test_parse_quadlet_file_malformed_line(tmp_path):
    """Verify ValueError on malformed line lacking equal sign."""
    bad_file = tmp_path / "bad.container"
    bad_file.write_text("[Unit]\nInvalidLineWithoutEquals\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed key-value line"):
        parse_quadlet_file(bad_file)


def test_parse_quadlet_file_comments_and_empty_lines(tmp_path):
    """Verify parser ignores comments and empty lines."""
    file_content = """# Comment line
; Semicolon comment

[Unit]
Description=Test Description # note

[Container]
ContainerName=test
"""
    tmp_file = tmp_path / "comments.container"
    tmp_file.write_text(file_content, encoding="utf-8")
    parsed = parse_quadlet_file(tmp_file)
    assert "Unit" in parsed
    assert parsed["Unit"]["Description"] == ["Test Description # note"]
    assert parsed["Container"]["ContainerName"] == ["test"]


def test_validate_quadlet_config_missing_sections():
    """Verify ValueError when required sections are missing."""
    config = {"Unit": {}, "Container": {}}
    with pytest.raises(ValueError, match="Missing required sections"):
        validate_quadlet_config(config)


def test_validate_quadlet_config_missing_volume():
    """Verify ValueError when persistent history volume is omitted."""
    config = {
        "Unit": {},
        "Container": {
            "ContainerName": ["production-benchmark-ui"],
            "Image": ["ghcr.io/sheepdestroyer/benchmark-ui:latest"],
            "PublishPort": ["127.0.0.1:8501:8501"],
            "Label": ["wud.watch.digest=true"],
        },
        "Service": {},
        "Install": {},
    }
    with pytest.raises(ValueError, match="Missing required persistent history volume"):
        validate_quadlet_config(config)


def test_validate_quadlet_config_missing_publish_port():
    """Verify ValueError when PublishPort is omitted."""
    config = {
        "Unit": {},
        "Container": {
            "ContainerName": ["production-benchmark-ui"],
            "Image": ["ghcr.io/sheepdestroyer/benchmark-ui:latest"],
            "Volume": ["/mnt/DATA/boy/prod/benchmark-ui/data/history:/app/history:Z"],
            "Label": ["wud.watch.digest=true"],
        },
        "Service": {},
        "Install": {},
    }
    with pytest.raises(ValueError, match="Missing PublishPort configuration"):
        validate_quadlet_config(config)


def test_validate_quadlet_config_insecure_port():
    """Verify ValueError when PublishPort binds to public interface instead of loopback."""
    config = {
        "Unit": {},
        "Container": {
            "ContainerName": ["production-benchmark-ui"],
            "Image": ["ghcr.io/sheepdestroyer/benchmark-ui:latest"],
            "PublishPort": ["0.0.0.0:8501:8501"],
            "Volume": ["/mnt/DATA/boy/prod/benchmark-ui/data/history:/app/history:Z"],
            "Label": ["wud.watch.digest=true"],
        },
        "Service": {},
        "Install": {},
    }
    with pytest.raises(ValueError, match="PublishPort must bind to 127.0.0.1 loopback"):
        validate_quadlet_config(config)


def test_validate_quadlet_config_missing_container_name_or_image():
    """Verify ValueError when ContainerName or Image is missing."""
    config_no_name = {
        "Unit": {},
        "Container": {
            "Image": ["ghcr.io/sheepdestroyer/benchmark-ui:latest"],
            "PublishPort": ["127.0.0.1:8501:8501"],
            "Volume": ["/mnt/DATA/boy/prod/benchmark-ui/data/history:/app/history:Z"],
            "Label": ["wud.watch.digest=true"],
        },
        "Service": {},
        "Install": {},
    }
    with pytest.raises(ValueError, match="Missing ContainerName"):
        validate_quadlet_config(config_no_name)

    config_no_image = {
        "Unit": {},
        "Container": {
            "ContainerName": ["production-benchmark-ui"],
            "PublishPort": ["127.0.0.1:8501:8501"],
            "Volume": ["/mnt/DATA/boy/prod/benchmark-ui/data/history:/app/history:Z"],
            "Label": ["wud.watch.digest=true"],
        },
        "Service": {},
        "Install": {},
    }
    with pytest.raises(ValueError, match="Missing Image"):
        validate_quadlet_config(config_no_image)


def test_validate_quadlet_config_missing_wud_label():
    """Verify ValueError when WUD digest label is omitted."""
    config = {
        "Unit": {},
        "Container": {
            "ContainerName": ["production-benchmark-ui"],
            "Image": ["ghcr.io/sheepdestroyer/benchmark-ui:latest"],
            "PublishPort": ["127.0.0.1:8501:8501"],
            "Volume": ["/mnt/DATA/boy/prod/benchmark-ui/data/history:/app/history:Z"],
            "Label": [],
        },
        "Service": {},
        "Install": {},
    }
    with pytest.raises(ValueError, match="Missing Label=wud.watch.digest=true"):
        validate_quadlet_config(config)
