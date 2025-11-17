"""
Effect plugin discovery and loading system.

This module automatically discovers all effect plugins in the effects/ directory
and makes them available for use.
"""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type
from core.base_effect import BaseEffect


def discover_effects() -> Dict[str, Type[BaseEffect]]:
    """Discover all effect plugins in the effects directory

    Returns:
        Dictionary mapping effect names (e.g., "seasonal/christmas") to effect classes
    """
    effects = {}
    effects_dir = Path(__file__).parent

    # Scan all subdirectories
    for category_dir in effects_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('_'):
            continue

        category_name = category_dir.name

        # Scan all Python files in the category
        for effect_file in category_dir.glob('*.py'):
            if effect_file.name.startswith('_'):
                continue

            # Import the module
            module_name = f"effects.{category_name}.{effect_file.stem}"
            try:
                module = importlib.import_module(module_name)

                # Find all BaseEffect subclasses in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseEffect) and obj is not BaseEffect:
                        # Register with category/name path
                        effect_key = f"{category_name}/{effect_file.stem}"
                        effects[effect_key] = obj

            except Exception as e:
                print(f"Warning: Could not load effect from {module_name}: {e}")

    return effects


def list_effects() -> List[tuple]:
    """List all available effects with metadata

    Returns:
        List of tuples: (key, name, description, category)
    """
    effects = discover_effects()
    effect_list = []

    for key, effect_class in sorted(effects.items()):
        try:
            name = effect_class.get_name()
            description = effect_class.get_description()
            category = effect_class.get_category()
            effect_list.append((key, name, description, category))
        except:
            # Skip effects that don't properly implement the interface
            continue

    return effect_list


def get_effect_class(effect_key: str) -> Type[BaseEffect]:
    """Get an effect class by its key

    Args:
        effect_key: Effect key (e.g., "seasonal/christmas")

    Returns:
        The effect class

    Raises:
        KeyError: If effect is not found
    """
    effects = discover_effects()
    if effect_key not in effects:
        raise KeyError(f"Effect '{effect_key}' not found. Available effects: {list(effects.keys())}")

    return effects[effect_key]
