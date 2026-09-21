"""Shared W&B configuration helpers for experiment entry points."""

import os


_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def wandb_enabled(default=True):
    """Return whether scripts should enable their logger's W&B integration."""
    value = os.getenv("POTION_WANDB")
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        "POTION_WANDB must be one of: 1, 0, true, false, yes, no, on, off"
    )


def make_wandb_kwargs(default_project, name, group, tags, config):
    """Build logger init options without importing or calling W&B."""
    kwargs = {
        "project": os.getenv("WANDB_PROJECT", default_project),
        "name": name,
        "group": group,
        "tags": tuple(tags),
        "config": dict(config),
    }
    entity = os.getenv("WANDB_ENTITY")
    if entity:
        kwargs["entity"] = entity
    return kwargs
