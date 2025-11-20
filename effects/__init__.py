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

    # Also discover user pipelines
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
    """Create a dynamic effect class for a saved user pipeline

    Args:
        pipeline_key: The pipeline key (e.g., "user_mypipeline")
        config: The pipeline configuration dictionary

    Returns:
        A dynamically created effect class
    """
    import tkinter as tk
    from tkinter import ttk
    import numpy as np

    class UserPipelineEffect(BaseUIEffect):
        """Dynamically created user pipeline effect"""

        # Store config at class level
        _pipeline_config = config
        _pipeline_key = pipeline_key

        def __init__(self, width, height, root=None):
            super().__init__(width, height, root)

            self.width = width
            self.height = height
            self.root = root

            # Load effects from config
            self.effects = []
            self._load_effects_from_config()

        def _load_effects_from_config(self):
            """Load and instantiate effects from the pipeline config"""
            for effect_config in self._pipeline_config.get('effects', []):
                try:
                    module_name = effect_config['module']
                    class_name = effect_config['class_name']

                    # Import the effect module
                    module = importlib.import_module(f'effects.opencv.{module_name}')
                    effect_class = getattr(module, class_name)

                    # Create instance
                    effect = effect_class(self.width, self.height, self.root)

                    # Apply saved parameters
                    for param_name, value in effect_config.get('params', {}).items():
                        if hasattr(effect, param_name):
                            attr = getattr(effect, param_name)
                            if isinstance(attr, tk.Variable):
                                try:
                                    attr.set(value)
                                except:
                                    pass

                    self.effects.append(effect)

                except Exception as e:
                    print(f"Warning: Could not load effect from pipeline: {e}")

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

        def create_control_panel(self, parent):
            """Create control panel showing all effect controls"""
            self.control_panel = ttk.Frame(parent)

            padding = {'padx': 10, 'pady': 5}

            # Header
            header_frame = ttk.Frame(self.control_panel)
            header_frame.pack(fill='x', **padding)

            title_label = ttk.Label(
                header_frame,
                text=self.get_name(),
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(side='left')

            # Edit button to open in Pipeline Builder
            def on_edit():
                # Signal restart - main loop will read _pipeline_key from effect
                if self.root:
                    self.root.event_generate('<<EditPipeline>>')

            edit_btn = ttk.Button(
                header_frame,
                text="Edit",
                command=on_edit,
                width=6
            )
            edit_btn.pack(side='right')

            edit_note = ttk.Label(
                header_frame,
                text="(restarts app)",
                font=('TkDefaultFont', 8, 'italic')
            )
            edit_note.pack(side='right', padx=(0, 5))

            # Add each effect's control panel
            for effect in self.effects:
                if hasattr(effect, 'create_control_panel'):
                    effect_panel = effect.create_control_panel(self.control_panel)
                    if effect_panel:
                        effect_panel.pack(fill='x', padx=5, pady=5)

            return self.control_panel

        def draw(self, frame, face_mask=None):
            """Apply all effects in the pipeline"""
            result = frame

            for effect in self.effects:
                if hasattr(effect, 'enabled'):
                    if not effect.enabled.get():
                        continue

                if hasattr(effect, 'update'):
                    effect.update()

                result = effect.draw(result, face_mask)

            return result

        def cleanup(self):
            """Cleanup all effects"""
            for effect in self.effects:
                if hasattr(effect, 'cleanup'):
                    effect.cleanup()

    # Create a unique class name
    class_name = f"UserPipeline_{pipeline_key.replace('user_', '')}"
    UserPipelineEffect.__name__ = class_name
    UserPipelineEffect.__qualname__ = class_name

    return UserPipelineEffect


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
