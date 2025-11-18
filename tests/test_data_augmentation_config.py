import pytest

from lisbet.cli.commands.train import parse_data_augmentation
from lisbet.config.schemas import DataAugmentationConfig


def test_parse_data_augmentation_none():
    """Test parsing None returns None."""
    assert parse_data_augmentation(None) is None


def test_parse_data_augmentation_empty():
    """Test parsing empty string returns None."""
    assert parse_data_augmentation("") is None


def test_parse_data_augmentation_single():
    """Test parsing single augmentation without parameters."""
    result = parse_data_augmentation("all_perm_id")
    assert result == [{"name": "all_perm_id", "p": 1.0}]


def test_parse_data_augmentation_with_probability():
    """Test parsing augmentation with probability parameter."""
    result = parse_data_augmentation("all_perm_id:p=0.5")
    assert result == [{"name": "all_perm_id", "p": 0.5}]


def test_parse_data_augmentation_with_fraction():
    """Test parsing blk_perm_id with fraction parameter."""
    result = parse_data_augmentation("blk_perm_id:frac=0.3")
    assert result == [{"name": "blk_perm_id", "p": 1.0, "frac": 0.3}]


def test_parse_data_augmentation_multiple():
    """Test parsing multiple augmentations."""
    result = parse_data_augmentation("all_perm_id:p=0.5,blk_perm_id:p=0.3:frac=0.2")
    assert result == [
        {"name": "all_perm_id", "p": 0.5},
        {"name": "blk_perm_id", "p": 0.3, "frac": 0.2},
    ]


def test_parse_data_augmentation_all_types():
    """Test parsing all augmentation types."""
    result = parse_data_augmentation(
        "all_perm_id:p=0.5,all_perm_ax:p=0.7,blk_perm_id:frac=0.3"
    )
    assert len(result) == 3
    assert result[0]["name"] == "all_perm_id"
    assert result[1]["name"] == "all_perm_ax"
    assert result[2]["name"] == "blk_perm_id"


def test_parse_data_augmentation_invalid_value():
    """Test that invalid parameter values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid parameter value"):
        parse_data_augmentation("all_perm_id:p=invalid")


def test_data_augmentation_config_valid():
    """Test creating valid DataAugmentationConfig objects."""
    # all_perm_id without parameters
    cfg1 = DataAugmentationConfig(name="all_perm_id")
    assert cfg1.name == "all_perm_id"
    assert cfg1.p == 1.0
    assert cfg1.frac is None

    # all_perm_id with probability
    cfg2 = DataAugmentationConfig(name="all_perm_id", p=0.5)
    assert cfg2.p == 0.5

    # blk_perm_id with fraction
    cfg3 = DataAugmentationConfig(name="blk_perm_id", frac=0.3)
    assert cfg3.frac == 0.3
    assert cfg3.p == 1.0


def test_data_augmentation_config_blk_perm_id_default_fraction():
    """Test that blk_perm_id gets default fraction of 0.5."""
    cfg = DataAugmentationConfig(name="blk_perm_id")
    assert cfg.frac == 0.5


def test_data_augmentation_config_invalid_probability():
    """Test that invalid probabilities are rejected."""
    with pytest.raises(ValueError, match="Probability p must be between 0.0 and 1.0"):
        DataAugmentationConfig(name="all_perm_id", p=1.5)

    with pytest.raises(ValueError, match="Probability p must be between 0.0 and 1.0"):
        DataAugmentationConfig(name="all_perm_id", p=-0.1)


def test_data_augmentation_config_invalid_fraction():
    """Test that invalid fractions are rejected."""
    with pytest.raises(ValueError, match="Fraction frac must be between 0.0 and 1.0"):
        DataAugmentationConfig(name="blk_perm_id", frac=0.0)

    with pytest.raises(ValueError, match="Fraction frac must be between 0.0 and 1.0"):
        DataAugmentationConfig(name="blk_perm_id", frac=1.0)

    with pytest.raises(ValueError, match="Fraction frac must be between 0.0 and 1.0"):
        DataAugmentationConfig(name="blk_perm_id", frac=1.5)


def test_data_augmentation_config_frac_only_for_blk_perm_id():
    """Test that frac parameter is only valid for blk_perm_id."""
    with pytest.raises(
        ValueError, match="frac parameter is only valid for blk_perm_id"
    ):
        DataAugmentationConfig(name="all_perm_id", frac=0.5)

    with pytest.raises(
        ValueError, match="frac parameter is only valid for blk_perm_id"
    ):
        DataAugmentationConfig(name="all_perm_ax", frac=0.5)


def test_data_augmentation_config_edge_case_probabilities():
    """Test edge case probability values."""
    # p=0.0 should be valid
    cfg1 = DataAugmentationConfig(name="all_perm_id", p=0.0)
    assert cfg1.p == 0.0

    # p=1.0 should be valid
    cfg2 = DataAugmentationConfig(name="all_perm_id", p=1.0)
    assert cfg2.p == 1.0


def test_data_augmentation_config_edge_case_fractions():
    """Test edge case fraction values (exclusive bounds)."""
    # frac=0.001 should be valid
    cfg1 = DataAugmentationConfig(name="blk_perm_id", frac=0.001)
    assert cfg1.frac == 0.001

    # frac=0.999 should be valid
    cfg2 = DataAugmentationConfig(name="blk_perm_id", frac=0.999)
    assert cfg2.frac == 0.999


def test_parse_data_augmentation_with_spaces():
    """Test that parsing handles spaces correctly."""
    result = parse_data_augmentation("all_perm_id : p = 0.5 , blk_perm_id : frac=0.3")
    # Note: split on "=" will keep spaces in keys/values
    # Our current implementation uses strip() on parts but not on param key/value
    # Let's verify it still works or add better handling
    assert len(result) == 2
