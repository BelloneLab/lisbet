"""Utility for applying parameter overrides to configuration dataclasses.

This module provides a function to apply a dictionary of overrides to a dataclass
instance, returning a new instance with updated values. It is intended for use with
LISBET model configuration dataclasses.

The function is type-safe and only updates fields present in the dataclass. Unknown
keys are ignored.

Example
-------
from lisbet.config.presets import TransformerBaseConfig
from lisbet.config.overrides import apply_overrides

base_cfg = TransformerBaseConfig()
overrides = {"embedding_dim": 64, "num_layers": 8}
new_cfg = apply_overrides(base_cfg, overrides)
"""

import logging
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any, TypeVar

T = TypeVar("T")


def cast_value(value: str, target_type):
    """
    Convert a string value to the specified target type.
    This function handles basic type conversions like str, int, float, and bool.
    Parameters
    ----------
    value : str
        The string value to convert.
    target_type : type
        The type to which the value should be converted (e.g., int, float, bool, str).

    Returns
    -------
    Any
        The converted value in the specified type.
    """
    # Handle common types; expand as needed
    if target_type is bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return value
    # Add more sophisticated parsing for lists, dicts, etc., if needed
    return value


def apply_overrides(config: T, overrides: Mapping[str, Any]) -> T:
    """
    Return a new config dataclass with specified fields overridden.

    Parameters
    ----------
    config : T
        The original dataclass instance.
    overrides : Mapping[str, Any]
        Dictionary of field names and new values to override.

    Returns
    -------
    T
        A new dataclass instance with the specified fields updated.

    Notes
    -----
    - Only fields present in the dataclass are updated.
    - Unknown keys in overrides are ignored.
    """
    valid_fields = {f.name: f.type for f in fields(config)}
    filtered = {}
    for k, v in overrides.items():
        if k in valid_fields:
            field_type = valid_fields[k]
            # Cast the value to the appropriate type
            filtered[k] = cast_value(v, field_type)
        else:
            logging.warning(f"Unknown field '{k}' in overrides; ignoring.")
    return replace(config, **filtered)
