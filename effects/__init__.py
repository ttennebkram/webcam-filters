"""
Effect plugin discovery and loading system.

This module automatically discovers all effect plugins in the effects/ directory
and makes them available for use.
"""

import os
import importlib
import inspect
import json
from pathlib import Path
from typing import Dict, List, Type
from core.base_effect import BaseEffect, BaseUIEffect


def _restore_tk_variables(obj, data):
    """Recursively restore tk.Variable values from serialized data.

    Handles:
    - Direct tk.Variable attributes
    - Dicts containing tk.Variables or nested dicts/lists
    - Lists containing tk.Variables or nested dicts/lists
    """
    import tkinter as tk

    if data is None:
        return

    if isinstance(obj, tk.Variable):
        try:
            obj.set(data)
        except:
            pass
    elif isinstance(obj, dict) and isinstance(data, dict):
        for key, value in data.items():
            if key in obj:
                _restore_tk_variables(obj[key], value)
    elif isinstance(obj, list) and isinstance(data, list):
        for i, value in enumerate(data):
            if i < len(obj):
                _restore_tk_variables(obj[i], value)


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

    # Also discover user pipelines (just adds keys to dropdown, loaded via pipeline_builder2)
    user_pipelines = discover_user_pipelines()
    effects.update(user_pipelines)

    return effects


def discover_user_pipelines() -> Dict[str, Type[BaseEffect]]:
    """Discover saved user pipelines and create effect classes for them

    Returns:
        Dictionary mapping pipeline names (e.g., "opencv/user_mypipeline") to effect classes
    """
    pipelines = {}

    # Get pipelines directory (in project root)
    effects_dir = Path(__file__).parent
    project_root = effects_dir.parent
    pipelines_dir = project_root / 'pipelines'

    if not pipelines_dir.exists():
        return pipelines

    # Find all pipeline JSON files
    for pipeline_file in pipelines_dir.glob('*.json'):
        name = pipeline_file.stem  # filename without extension
        try:
            with open(pipeline_file, 'r') as f:
                config = json.load(f)

            # Create a dynamic class for this pipeline
            pipeline_key = f"user_{name}"
            pipeline_class = create_user_pipeline_class(pipeline_key, config)
            if pipeline_class:
                effect_key = f"opencv/{pipeline_key}"
                pipelines[effect_key] = pipeline_class
        except Exception as e:
            print(f"Warning: Could not load pipeline {pipeline_file}: {e}")

    return pipelines


def create_user_pipeline_class(pipeline_key: str, config: dict) -> Type[BaseEffect]:
    """Create a minimal placeholder class for a saved user pipeline.

    This class only provides metadata for the dropdown list.
    Actual rendering is handled by pipeline_builder2 when selected.

    Args:
        pipeline_key: The pipeline key (e.g., "user_mypipeline")
        config: The pipeline configuration dictionary

    Returns:
        A minimal effect class with just metadata
    """

    class UserPipelinePlaceholder(BaseEffect):
        """Placeholder class for user pipeline - actual rendering via pipeline_builder2"""

        # Store config at class level
        _pipeline_config = config
        _pipeline_key = pipeline_key

        def __init__(self, width, height):
            # This should never be instantiated - main.py redirects to pipeline_builder2
            raise RuntimeError(
                f"UserPipelinePlaceholder should not be instantiated. "
                f"Select '{pipeline_key}' from dropdown to load via pipeline_builder2."
            )

        @classmethod
        def get_name(cls) -> str:
            return cls._pipeline_config.get('name', cls._pipeline_key)

        @classmethod
        def get_description(cls) -> str:
            num_effects = len(cls._pipeline_config.get('effects', []))
            return f"User pipeline with {num_effects} effects"

        @classmethod
        def get_category(cls) -> str:
            return "opencv"

        def draw(self, frame, face_mask=None):
            return frame

    # Create a unique class name
    class_name = f"UserPipeline_{pipeline_key.replace('user_', '')}"
    UserPipelinePlaceholder.__name__ = class_name
    UserPipelinePlaceholder.__qualname__ = class_name

    return UserPipelinePlaceholder


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
