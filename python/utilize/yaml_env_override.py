# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""Override YAML config values from KSANA__* environment variables.

The original YAML is read in place (comments and formatting are preserved via
ruamel.yaml round-trip). Only the fields explicitly targeted by KSANA__*
environment variables are modified; every other value keeps its original YAML
content. The result is written to a new file.

Environment variable naming convention
--------------------------------------
Prefix: ``KSANA__``
Path separator: ``__`` (double underscore)

Each segment between ``__`` is one YAML key. A single ``_`` inside a segment is
kept as part of the key name, so keys that contain underscores do not collide
with nesting.

Examples::

    KSANA__SETTING__BLOCK_MANAGER__BLOCK_HOST_MEMORY_FACTOR=0.01
      -> setting.block_manager.block_host_memory_factor = 0.01

    KSANA__SETTING__BATCH_SCHEDULER__ENABLE_AUTO_PREFIX_CACHE=false
      -> setting.batch_scheduler.enable_auto_prefix_cache = false

Keys are lower-cased to match the YAML style used by KsanaLLM.

Type handling
-------------
When the target key already exists in the YAML, the env string is coerced to the
*existing* value's type (bool/int/float/str). This avoids surprises such as a
version string ``"1.10"`` being turned into the float ``1.1``. When the key does
not exist yet, the value is auto-detected (bool/int/float/null/JSON/str).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from ruamel.yaml import YAML
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt

ENV_PREFIX = "KSANA__"
PATH_SEPARATOR = "__"

_TRUE_LITERALS = ("true", "yes", "on")
_FALSE_LITERALS = ("false", "no", "off")
_NULL_LITERALS = ("null", "none", "~")


def _make_yaml() -> YAML:
    """Create a round-trip YAML handler that preserves comments and layout."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # avoid line wrapping of long scalars
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _auto_parse(text: str) -> Any:
    """Auto-detect the type of an env string (used for brand-new keys)."""
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in _TRUE_LITERALS:
        return True
    if lowered in _FALSE_LITERALS:
        return False
    if lowered in _NULL_LITERALS:
        return None

    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        if any(ch in stripped for ch in (".", "e", "E")):
            return float(stripped)
        return int(stripped)
    except ValueError:
        pass

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped


def _coerce_to_type(text: str, template: Any) -> Any:
    """Coerce ``text`` to the type of ``template`` (the existing YAML value)."""
    stripped = text.strip()

    # bool must be checked before int, since bool is a subclass of int.
    if isinstance(template, (bool, ScalarBoolean)):
        lowered = stripped.lower()
        if lowered in _TRUE_LITERALS:
            return True
        if lowered in _FALSE_LITERALS:
            return False
        raise ValueError(f"Cannot parse {text!r} as bool")

    if isinstance(template, (int, ScalarInt)) and not isinstance(template, bool):
        return int(stripped)

    if isinstance(template, (float, ScalarFloat)):
        return float(stripped)

    if isinstance(template, str):
        return stripped

    if template is None:
        # Existing value is null: fall back to auto-detection.
        return _auto_parse(text)

    # Existing value is a structured type (list/dict) or unknown: parse loosely.
    return _auto_parse(text)


def env_name_to_path(env_name: str, prefix: str = ENV_PREFIX) -> Optional[Tuple[str, ...]]:
    """Convert ``KSANA__SETTING__FOO_BAR`` to ``('setting', 'foo_bar')``."""
    if not env_name.startswith(prefix):
        return None
    remainder = env_name[len(prefix):]
    if not remainder:
        return None
    segments = [seg.lower() for seg in remainder.split(PATH_SEPARATOR) if seg]
    if not segments:
        return None
    return tuple(segments)


def collect_ksana_env_overrides(
    environ: Optional[Mapping[str, str]] = None,
    prefix: str = ENV_PREFIX,
) -> Dict[Tuple[str, ...], str]:
    """Collect raw ``KSANA__*`` overrides (path tuple -> raw string value)."""
    env = os.environ if environ is None else environ
    overrides: Dict[Tuple[str, ...], str] = {}
    for name, raw in env.items():
        path = env_name_to_path(name, prefix=prefix)
        if path is None:
            continue
        overrides[path] = raw
    return overrides


def _apply_single(config: MutableMapping[str, Any], path: Tuple[str, ...], raw: str) -> Any:
    """Set one override on ``config`` in place, coercing to the existing type.

    Returns the value actually written.
    """
    cursor: Any = config
    for key in path[:-1]:
        child = cursor.get(key) if isinstance(cursor, Mapping) else None
        if not isinstance(child, MutableMapping):
            child = {}
            cursor[key] = child
        cursor = child

    leaf_key = path[-1]
    if isinstance(cursor, Mapping) and leaf_key in cursor:
        value = _coerce_to_type(raw, cursor[leaf_key])
    else:
        value = _auto_parse(raw)
    cursor[leaf_key] = value
    return value


def apply_env_overrides(
    config: MutableMapping[str, Any],
    environ: Optional[Mapping[str, str]] = None,
    prefix: str = ENV_PREFIX,
) -> Dict[str, Any]:
    """Apply all ``KSANA__*`` env overrides onto ``config`` in place.

    Returns a human-readable ``{"a.b.c": value}`` map of applied overrides.
    Shorter paths are applied first so nested overrides win over ancestors.
    """
    applied: Dict[str, Any] = {}
    overrides = collect_ksana_env_overrides(environ, prefix=prefix)
    for path in sorted(overrides, key=len):
        value = _apply_single(config, path, overrides[path])
        applied[".".join(path)] = value
    return applied


def override_yaml_file(
    input_path: str,
    output_path: str,
    environ: Optional[Mapping[str, str]] = None,
    prefix: str = ENV_PREFIX,
) -> Dict[str, Any]:
    """Read ``input_path`` (comments preserved), apply env overrides, write to
    ``output_path``. Values without a matching env var keep the original YAML.

    Returns the applied overrides map.
    """
    yaml = _make_yaml()
    with open(input_path, "r", encoding="utf-8") as f:
        config = yaml.load(f)
    if config is None:
        config = {}

    applied = apply_env_overrides(config, environ=environ, prefix=prefix)

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Completely rewrite the output on every run: drop any pre-existing file so
    # repeated executions never leave stale content behind.
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return applied


def default_input_path() -> str:
    """Default input YAML: ``ksana_llm.yaml`` next to this script.

    Matches the packaged layout where both files live under
    ``site-packages/ksana_llm/``.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ksana_llm.yaml")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a commented YAML config, override values from KSANA__* "
            "environment variables, and write a new YAML file. Comments and "
            "un-overridden values are preserved. Use '__' as the path separator "
            "so keys that contain '_' are unambiguous."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default=default_input_path(),
        help="Input YAML config path. Defaults to 'ksana_llm.yaml' next to this script.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output YAML path (a new file; must differ from --input).",
    )
    parser.add_argument(
        "--prefix",
        default=ENV_PREFIX,
        help=f"Environment variable prefix (default: {ENV_PREFIX}).",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("error: --output must be a different file from --input.", file=sys.stderr)
        return 2

    applied = override_yaml_file(args.input, args.output, prefix=args.prefix)

    if applied:
        print("Applied KSANA env overrides:")
        for path, value in applied.items():
            print(f"  {path} = {value!r}")
    else:
        print("No KSANA__* environment variables found; config copied unchanged.")
    print(f"Wrote config to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
