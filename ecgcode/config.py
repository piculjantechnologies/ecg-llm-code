"""Load the TOML run configuration and apply validated `section.key=value` overrides."""
import copy
import json
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"

# Allowed values for settings where a typo would silently change the experiment.
ENUMS = {
    ("model", "effort"): {"low", "medium", "high", "xhigh", "max"},
    ("model", "thinking_display"): {"summarized", "omitted"},
    ("model", "code_tool"): {"code_execution_20260521", "code_execution_20260120"},
    ("output", "aggregate"): {"max", "noisy_or"},
}


def _parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _check_type(where, default, value):
    if isinstance(default, bool):
        ok = isinstance(value, bool)
        hint = "true or false (lowercase)"
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
        hint = "an integer"
    elif isinstance(default, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        hint = "a number"
    elif isinstance(default, str):
        ok = isinstance(value, str)
        hint = "a string"
    else:
        ok, hint = True, ""
    if not ok:
        raise ValueError(f"--set {where}: expected {hint}, got {value!r}")


def load_config(path=None, overrides=()):
    with open(path or DEFAULT_CONFIG, "rb") as f:
        cfg = copy.deepcopy(tomllib.load(f))
    for item in overrides:
        key, sep, raw = item.partition("=")
        section, _, name = key.strip().partition(".")
        if not sep or not name:
            raise ValueError(f"--set must look like section.key=value, got {item!r}")
        if section not in cfg or name not in cfg[section]:
            raise ValueError(f"--set {key}: unknown setting (check spelling against config/default.toml)")
        value = _parse_value(raw)
        _check_type(key, cfg[section][name], value)
        allowed = ENUMS.get((section, name))
        if allowed and value not in allowed:
            raise ValueError(f"--set {key}: must be one of {sorted(allowed)}, got {value!r}")
        cfg[section][name] = float(value) if isinstance(cfg[section][name], float) else value
    return cfg


def config_diff(cfg, base=None):
    """Settings that differ from the default config, as 'section.key: default -> value'."""
    base = base or load_config()
    out = []
    for section, values in cfg.items():
        for name, value in values.items():
            default = base.get(section, {}).get(name, "<missing>")
            if value != default:
                out.append(f"{section}.{name}: {default!r} -> {value!r}")
    return out


def project_path(relative):
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
