"""Runtime configuration shared by Foilage's SU2 entry points."""

import configparser
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "foilage.cfg"


def _resolve_path(value: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    path = Path(expanded)
    return path if path.is_absolute() else (base_dir / path).resolve()


def resolve_su2_executable(config_path=None) -> Path:
    """Resolve SU2_CFD from foilage.cfg without bundling the solver."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Foilage configuration not found: {path}. "
            "Copy foilage.cfg.example to foilage.cfg and set [su2].root."
        )

    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_section("su2"):
        raise ValueError(f"{path} is missing the [su2] section")

    if parser.has_option("su2", "executable"):
        executable = _resolve_path(parser.get("su2", "executable"), path.parent)
    elif parser.has_option("su2", "root"):
        root = _resolve_path(parser.get("su2", "root"), path.parent)
        executable_name = "SU2_CFD.exe" if os.name == "nt" else "SU2_CFD"
        executable = root / "bin" / executable_name
    else:
        raise ValueError(f"{path} needs either [su2] root or executable")

    return executable
