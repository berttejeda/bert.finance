"""Plugin base class — all exporter plugins must inherit from this."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PluginBase(ABC):
    """Standardized entrypoint for exporter plugins.

    Subclasses must implement:
        name       — unique plugin identifier (must match config name)
        run(args)  — main logic, receives the ``args`` dict from config.yaml

    Plugins can optionally override ``tickers`` in their args to use a
    different ticker list than the global one.  If ``args.tickers`` is not
    set, the loader will inject the global ticker list automatically.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"plugin.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name (matches the ``name`` field in config.yaml)."""
        ...

    @abstractmethod
    def run(self, args: Dict[str, Any]) -> None:
        """Execute the plugin's main logic.

        Args:
            args: The ``args`` dict defined under this plugin in config.yaml.
                  The loader guarantees the following keys are present:
                  - ``tickers``: List[str] of ticker symbols
                  - ``influxdb``: Dict with url, token, org, bucket
        """
        ...
