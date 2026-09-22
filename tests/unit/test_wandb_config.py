import pytest

from potion.evaluation.wandb_config import make_wandb_kwargs, wandb_enabled


def test_wandb_enabled_defaults_to_true_for_experiment_scripts(monkeypatch):
    monkeypatch.delenv("POTION_WANDB", raising=False)

    assert wandb_enabled()
    assert not wandb_enabled(default=False)


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_wandb_enabled_accepts_true_values(monkeypatch, value):
    monkeypatch.setenv("POTION_WANDB", value)

    assert wandb_enabled()


@pytest.mark.parametrize("value", ["0", "false", "NO", "off"])
def test_wandb_enabled_accepts_false_values(monkeypatch, value):
    monkeypatch.setenv("POTION_WANDB", value)

    assert not wandb_enabled()


def test_wandb_enabled_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("POTION_WANDB", "sometimes")

    with pytest.raises(ValueError, match="POTION_WANDB"):
        wandb_enabled()


def test_make_wandb_kwargs_uses_environment_overrides(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "custom-project")
    monkeypatch.setenv("WANDB_ENTITY", "research-team")

    kwargs = make_wandb_kwargs(
        default_project="default-project",
        name="run-name",
        group="run-group",
        tags=("one", "two"),
        config={"seed": 42},
    )

    assert kwargs == {
        "project": "custom-project",
        "entity": "research-team",
        "name": "run-name",
        "group": "run-group",
        "tags": ("one", "two"),
        "config": {"seed": 42},
    }


def test_make_wandb_kwargs_uses_default_project(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    kwargs = make_wandb_kwargs(
        default_project="default-project",
        name="run-name",
        group="run-group",
        tags=(),
        config={},
    )

    assert kwargs["project"] == "default-project"
    assert "entity" not in kwargs
