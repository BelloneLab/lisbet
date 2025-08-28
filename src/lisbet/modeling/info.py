import pprint

import yaml
from rich.console import Console
from rich.table import Table


def model_info(model_path):
    """Print information about a LISBET model config file."""

    with open(model_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    console = Console()
    table = Table(title="LISBET Model Configuration")

    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in config.items():
        # Pretty-print nested dicts/lists
        if isinstance(value, (dict, list)):
            value_str = pprint.pformat(value, compact=True, width=60)
        else:
            value_str = str(value)
        table.add_row(str(key), value_str)

    console.print(table)
