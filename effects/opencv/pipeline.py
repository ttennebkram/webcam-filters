"""
Pipeline base class for stacking multiple OpenCV effects.

Allows chaining effects together with a unified UI and configuration persistence.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import json
import os
from pathlib import Path
from core.base_effect import BaseUIEffect
import importlib


class BasePipelineEffect(BaseUIEffect):
    """Base class for pipeline effects that stack multiple OpenCV operations"""

    # Subclasses should override this with list of effect module names
    # e.g., ['blur', 'grayscale', 'threshold_simple']
    PIPELINE_EFFECTS = []

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Load effect classes and instantiate them
        self.effects = []
        self._load_effects(width, height, root)

        # Configuration file path (subclasses set this via get_config_filename)
        self.config_dir = Path(__file__).parent / 'configs'
        self.config_dir.mkdir(exist_ok=True)

        # Preferred window height (can be loaded from config)
        self._preferred_window_height = 800  # Default

        # Pre-load window height from config if it exists
        self._preload_window_height()

    def _load_effects(self, width, height, root):
        """Load and instantiate all effects in the pipeline"""
        for effect_name in self.PIPELINE_EFFECTS:
            try:
                # Import the effect module from opencv directory
                module = importlib.import_module(f"effects.opencv.{effect_name}")

                # Find the effect class (first BaseUIEffect subclass)
                effect_class = None
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) and
                        issubclass(obj, BaseUIEffect) and
                        obj is not BaseUIEffect and
                        not name.startswith('Base')):
                        effect_class = obj
                        break

                if effect_class:
                    effect = effect_class(width, height, root)
                    self.effects.append(effect)
                else:
                    print(f"Warning: No effect class found in {effect_name}")

            except Exception as e:
                print(f"Warning: Could not load effect '{effect_name}': {e}")

    @classmethod
    def get_config_filename(cls) -> str:
        """Return the configuration filename for this pipeline.

        Subclasses should override this to return their specific config name.
        e.g., 'blobs' will save to 'blobs.json'
        """
        return "pipeline"

    @classmethod
    def get_name(cls) -> str:
        return "Pipeline"

    @classmethod
    def get_description(cls) -> str:
        return "Stack multiple OpenCV effects in sequence"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_preferred_window_height(self):
        """Return preferred window height (may be loaded from config)"""
        return self._preferred_window_height

    def _preload_window_height(self):
        """Load just the window height from config before UI is created"""
        config_path = self._get_config_path()
        print(f"DEBUG: Preloading window height from {config_path}")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                if '_window_height' in settings:
                    self._preferred_window_height = settings['_window_height']
                    print(f"DEBUG: Loaded preferred height: {self._preferred_window_height}")
            except Exception as e:
                print(f"DEBUG: Error loading config: {e}")
        else:
            print(f"DEBUG: Config file does not exist")

    def create_control_panel(self, parent):
        """Create scrollable Tkinter control panel with all stacked effect UIs"""
        self.control_panel = ttk.Frame(parent)

        # Create canvas for scrolling
        canvas = tk.Canvas(self.control_panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.control_panel, orient='vertical', command=canvas.yview)

        # Scrollable frame inside canvas
        self.scrollable_frame = ttk.Frame(canvas)

        # Configure scroll region when frame size changes
        self.scrollable_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )

        # Create window in canvas
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        canvas.bind_all('<MouseWheel>', _on_mousewheel)

        # Pack canvas and scrollbar
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Store canvas reference for cleanup
        self._canvas = canvas

        # Add save warning message at the top (use tk.Label for color support)
        warning_label = tk.Label(
            self.scrollable_frame,
            text="Note: These values are not persisted unless you click Save Configuration",
            font=('TkDefaultFont', 10, 'italic'),
            fg='red'
        )
        warning_label.pack(anchor='w', padx=10, pady=(5, 10))

        # Add each effect's control panel
        for i, effect in enumerate(self.effects):
            # Create frame for this effect
            effect_frame = ttk.Frame(self.scrollable_frame)
            effect_frame.pack(fill='x', padx=5, pady=5)

            # Add separator between effects (except before first)
            if i > 0:
                ttk.Separator(effect_frame, orient='horizontal').pack(fill='x', pady=(0, 10))

            # Create the effect's control panel
            effect.create_control_panel(effect_frame)
            effect.control_panel.pack(fill='x')

        # Bottom buttons frame
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill='x', padx=10, pady=15)

        # Reset to Defaults button
        reset_btn = ttk.Button(
            button_frame,
            text="Reset to Defaults",
            command=self._reset_to_defaults
        )
        reset_btn.pack(side='left', padx=5)

        # Save Configuration button
        save_btn = ttk.Button(
            button_frame,
            text="Save Configuration",
            command=self._save_configuration
        )
        save_btn.pack(side='left', padx=5)

        # Load saved configuration if it exists
        self._load_configuration()

        return self.control_panel

    def _get_config_path(self):
        """Get the full path to the configuration file"""
        filename = self.get_config_filename()
        return self.config_dir / f"{filename}.json"

    def _get_defaults_path(self):
        """Get the full path to the defaults file"""
        filename = self.get_config_filename()
        return self.config_dir / f"{filename}_defaults.json"

    def _collect_settings(self):
        """Collect all settings from all effects into a dictionary"""
        settings = {}

        for i, effect in enumerate(self.effects):
            effect_settings = {}
            effect_name = self.PIPELINE_EFFECTS[i]

            # Collect all tk variables from the effect
            for attr_name in dir(effect):
                attr = getattr(effect, attr_name)
                if isinstance(attr, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar)):
                    effect_settings[attr_name] = attr.get()

            settings[effect_name] = effect_settings

        return settings

    def _apply_settings(self, settings):
        """Apply settings dictionary to all effects"""
        for i, effect in enumerate(self.effects):
            effect_name = self.PIPELINE_EFFECTS[i]

            if effect_name not in settings:
                continue

            effect_settings = settings[effect_name]

            # Apply settings to tk variables
            for attr_name, value in effect_settings.items():
                if hasattr(effect, attr_name):
                    attr = getattr(effect, attr_name)
                    if isinstance(attr, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar)):
                        try:
                            attr.set(value)
                        except:
                            pass

            # Update UI labels if they exist
            self._update_effect_labels(effect)

    def _update_effect_labels(self, effect):
        """Update UI labels to reflect current values"""
        # Update common label patterns
        if hasattr(effect, 'ksize_x_label') and hasattr(effect, 'kernel_size_x'):
            effect.ksize_x_label.config(text=str(effect.kernel_size_x.get()))
        if hasattr(effect, 'ksize_y_label') and hasattr(effect, 'kernel_size_y'):
            effect.ksize_y_label.config(text=str(effect.kernel_size_y.get()))
        if hasattr(effect, 'sigma_label') and hasattr(effect, 'sigma_x'):
            sigma = effect.sigma_x.get()
            if sigma == 0:
                effect.sigma_label.config(text="0.0 (auto)")
            else:
                effect.sigma_label.config(text=f"{sigma:.1f}")
        if hasattr(effect, 'thresh_label') and hasattr(effect, 'thresh_value'):
            effect.thresh_label.config(text=str(effect.thresh_value.get()))
        if hasattr(effect, 'max_label') and hasattr(effect, 'max_value'):
            effect.max_label.config(text=str(effect.max_value.get()))
        if hasattr(effect, 'conv_combo') and hasattr(effect, 'conversion_index'):
            effect.conv_combo.current(effect.conversion_index.get())
        if hasattr(effect, 'type_combo') and hasattr(effect, 'thresh_type_index'):
            effect.type_combo.current(effect.thresh_type_index.get())

    def _save_configuration(self):
        """Save current configuration to JSON file"""
        settings = self._collect_settings()

        # Get current window height if available
        if hasattr(self, 'control_panel') and self.control_panel:
            try:
                # Get the toplevel window
                toplevel = self.control_panel.winfo_toplevel()
                window_height = toplevel.winfo_height()
                settings['_window_height'] = window_height
            except:
                pass

        config_path = self._get_config_path()

        try:
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def _load_configuration(self):
        """Load configuration from JSON file if it exists"""
        config_path = self._get_config_path()

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    settings = json.load(f)

                # Extract window height if saved
                if '_window_height' in settings:
                    self._preferred_window_height = settings['_window_height']
                    del settings['_window_height']

                self._apply_settings(settings)
                print(f"Configuration loaded from {config_path}")
            except Exception as e:
                print(f"Error loading configuration: {e}")

    def _reset_to_defaults(self):
        """Reset all effects to their default values"""
        # First check if there's a defaults file
        defaults_path = self._get_defaults_path()

        if defaults_path.exists():
            try:
                with open(defaults_path, 'r') as f:
                    settings = json.load(f)
                self._apply_settings(settings)
                print(f"Reset to defaults from {defaults_path}")
                return
            except Exception as e:
                print(f"Error loading defaults: {e}")

        # Otherwise, recreate effects with their built-in defaults
        width, height = self.width, self.height
        root = self.root_window

        for i, effect in enumerate(self.effects):
            effect_name = self.PIPELINE_EFFECTS[i]

            try:
                # Get the effect class
                module = importlib.import_module(f"effects.opencv.{effect_name}")
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) and
                        issubclass(obj, BaseUIEffect) and
                        obj is not BaseUIEffect and
                        not name.startswith('Base')):

                        # Create a temporary instance to get defaults
                        temp_effect = obj(width, height, root)

                        # Copy default values to current effect
                        for attr_name in dir(temp_effect):
                            attr = getattr(temp_effect, attr_name)
                            if isinstance(attr, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar)):
                                if hasattr(effect, attr_name):
                                    current_attr = getattr(effect, attr_name)
                                    if isinstance(current_attr, type(attr)):
                                        current_attr.set(attr.get())

                        # Update labels
                        self._update_effect_labels(effect)
                        break

            except Exception as e:
                print(f"Error resetting {effect_name}: {e}")

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply all effects in sequence"""
        result = frame

        for effect in self.effects:
            result = effect.draw(result, face_mask)

        return result

    def cleanup(self):
        """Clean up resources"""
        # Unbind mousewheel
        if hasattr(self, '_canvas'):
            self._canvas.unbind_all('<MouseWheel>')

        # Cleanup each effect
        for effect in self.effects:
            effect.cleanup()
