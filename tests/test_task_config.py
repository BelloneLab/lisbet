import pytest
from pydantic import ValidationError

from lisbet.config.schemas import TaskConfig, TrainingConfig


def _training_kwargs(**extra):
    base = {"epochs": 1, "batch_size": 2, "learning_rate": 1e-4}
    base.update(extra)
    return base


def test_task_configs_defaults_to_empty_dict():
    config = TrainingConfig.model_validate(_training_kwargs())
    assert config.task_configs == {}


def test_task_configs_coerces_string_values():
    """--set passes raw strings; pydantic coerces them to TaskConfig fields."""
    config = TrainingConfig.model_validate(
        _training_kwargs(task_configs={"geom": {"temperature": "0.2"}})
    )
    assert config.task_configs["geom"] == TaskConfig(temperature=0.2)


def test_task_config_temperature_defaults_to_none():
    assert TaskConfig().temperature is None


@pytest.mark.parametrize("value", [0, -1, -0.5])
def test_task_config_rejects_non_positive_temperature(value):
    with pytest.raises(ValidationError):
        TaskConfig(temperature=value)


def test_task_config_rejects_unknown_param():
    with pytest.raises(ValidationError):
        TaskConfig.model_validate({"temprature": 0.2})
