import re
from pathlib import Path
from typing import List

import pytest

from lisbet.datasets.dirtree import _find_tracking_files


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for testing."""
    return tmp_path


def create_files(temp_dir: Path, filenames: List[str], nested: bool = False) -> None:
    """Helper to create empty files with given names."""
    for filename in filenames:
        if nested and "/" in filename:
            # Create nested directory structure if path contains slashes
            full_path = temp_dir / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.touch()
        else:
            (temp_dir / filename).touch()


def test_single_csv(temp_dir):
    """Test when there's only one CSV file."""
    create_files(temp_dir, ["single.csv"])
    result = _find_tracking_files(temp_dir)
    assert result == temp_dir / "single.csv"


def test_no_csv_files(temp_dir):
    """Test when there are no CSV files."""
    create_files(temp_dir, ["file.txt", "data.xlsx"])
    with pytest.raises(ValueError, match="No tracking files found"):
        _find_tracking_files(temp_dir)


def test_dlc_single_match(temp_dir):
    """Test when DLC pattern matches exactly one file."""
    files = [
        "converted_Trial34DLC_HrnetW32_Nov12shuffle1_snapshot_195.csv",
        "other.csv",
        "another.csv",
    ]
    create_files(temp_dir, files)
    result = _find_tracking_files(temp_dir)
    assert result == temp_dir / files[0]


def test_dlc_multiple_matches(temp_dir):
    """Test when DLC pattern matches multiple files."""
    files = [
        "converted_Trial34DLC_HrnetW32_Nov12shuffle1.csv",
        "another_DLC_Dec15shuffle2.csv",
        "other.csv",
    ]
    create_files(temp_dir, files)
    with pytest.raises(ValueError, match="Multiple files match a DLC-like pattern"):
        _find_tracking_files(temp_dir)


def test_tracking_single_match(temp_dir):
    """Test when DLC pattern fails and tracking pattern matches exactly one file."""
    files = ["unmatched.csv", "mouse_tracking.csv", "other.csv"]
    create_files(temp_dir, files)
    result = _find_tracking_files(temp_dir)
    assert result == temp_dir / "mouse_tracking.csv"


def test_tracking_no_matches(temp_dir):
    """Test when both patterns fail to find matches."""
    files = ["unmatched1.csv", "unmatched2.csv"]
    create_files(temp_dir, files)
    with pytest.raises(ValueError, match="No tracking files found"):
        _find_tracking_files(temp_dir)


def test_tracking_multiple_matches(temp_dir):
    """Test when tracking pattern finds multiple matches."""
    files = ["mouse_tracking.csv", "rat_tracking.csv"]
    create_files(temp_dir, files)
    with pytest.raises(
        ValueError, match="Multiple files contain 'tracking' in their name"
    ):
        _find_tracking_files(temp_dir)


def test_nested_directory_search(temp_dir):
    """Test that the function finds files in nested directories."""
    files = [
        "subdir1/data.csv",
        "subdir2/subdir3/converted_Trial34DLC_HrnetW32_Nov12shuffle1.csv",
    ]
    create_files(temp_dir, files, nested=True)
    result = _find_tracking_files(temp_dir)
    assert result == temp_dir / files[1]


def test_empty_directory(temp_dir):
    """Test behavior with an empty directory."""
    with pytest.raises(ValueError, match="No tracking files found"):
        _find_tracking_files(temp_dir)


def test_dlc_various_months(temp_dir):
    """Test DLC pattern with different month formats."""
    months = [
        "converted_DLC_Jan12shuffle1.csv",
        "converted_DLC_Feb12shuffle1.csv",
        "converted_DLC_Mar12shuffle1.csv",
        "converted_DLC_Apr12shuffle1.csv",
        "converted_DLC_May12shuffle1.csv",
        "converted_DLC_Jun12shuffle1.csv",
        "converted_DLC_Jul12shuffle1.csv",
        "converted_DLC_Aug12shuffle1.csv",
        "converted_DLC_Sep12shuffle1.csv",
        "converted_DLC_Oct12shuffle1.csv",
        "converted_DLC_Nov12shuffle1.csv",
        "converted_DLC_Dec12shuffle1.csv",
    ]
    for month_file in months:
        create_files(temp_dir, [month_file])
        result = _find_tracking_files(temp_dir)
        assert result == temp_dir / month_file
        # Clean up for next iteration
        for file in temp_dir.glob("*.csv"):
            file.unlink()
