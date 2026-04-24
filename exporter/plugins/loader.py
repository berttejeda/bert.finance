"""Discover and load exporter plugins from config.yaml."""

import importlib
import logging
from typing import Any, Dict, List, Tuple

from plugins.base import PluginBase

logger = logging.getLogger("plugin.loader")


def _import_plugin(plugin_name: str) -> PluginBase:
    """Import ``plugins.<name>.plugin`` and return an instance of its Plugin class.

    Convention: each plugin directory must contain a ``plugin.py`` that
    exposes a class named ``Plugin`` inheriting from ``PluginBase``.
    """
    module_path = f"plugins.{plugin_name}.plugin"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Could not import plugin module '{module_path}': {e}"
        ) from e

    plugin_cls = getattr(module, "Plugin", None)
    if plugin_cls is None:
        raise ImportError(
            f"Plugin module '{module_path}' must export a class named 'Plugin'"
        )
    if not issubclass(plugin_cls, PluginBase):
        raise TypeError(
            f"Plugin class in '{module_path}' must inherit from PluginBase"
        )

    return plugin_cls()


def load_plugins(
    config: dict,
) -> List[Tuple[PluginBase, Dict[str, Any]]]:
    """Load all enabled plugins defined in config.

    For each plugin, the loader merges the global ``influxdb`` and ``tickers``
    settings into the plugin's ``args`` dict (plugin-level values take
    precedence).

    Args:
        config: The full parsed config dict (from config.yaml).

    Returns:
        List of (plugin_instance, args_dict) tuples for enabled plugins.
    """
    plugin_defs = config.get("plugins") or []
    loaded: List[Tuple[PluginBase, Dict[str, Any]]] = []

    global_influx = config.get("influxdb") or {}
    global_tickers = config.get("tickers") or []

    for entry in plugin_defs:
        name = entry.get("name")
        enabled = entry.get("enabled", True)
        args = dict(entry.get("args") or {})

        if not name:
            logger.warning("Skipping plugin entry with no 'name'")
            continue
        if not enabled:
            logger.info(f"Plugin '{name}' is disabled, skipping")
            continue

        # Inject global tickers if the plugin doesn't specify its own
        if "tickers" not in args:
            args["tickers"] = list(global_tickers)

        # Inject global influxdb config if the plugin doesn't specify its own
        if "influxdb" not in args:
            args["influxdb"] = dict(global_influx)

        try:
            plugin = _import_plugin(name)
            loaded.append((plugin, args))
            logger.info(f"Loaded plugin '{name}' (v{entry.get('version', '?')})")
        except (ImportError, TypeError) as e:
            logger.error(f"Failed to load plugin '{name}': {e}")

    return loaded
