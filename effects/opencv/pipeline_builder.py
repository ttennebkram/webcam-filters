"""
Pipeline Builder effect that allows dynamic creation of effect chains.

Provides UI to add/remove/reorder OpenCV effects at runtime.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from core.base_effect import BaseUIEffect


def get_opencv_effects():
    """Get list of available OpenCV effects (excluding pipeline classes)"""
    import importlib
    import pkgutil

    effects = []
    effects_dir = os.path.dirname(__file__)

    for importer, modname, ispkg in pkgutil.iter_modules([effects_dir]):
        # Skip pipeline-related modules
        if modname in ('pipeline', 'pipeline_builder'):
            continue
        try:
            module = importlib.import_module(f'effects.opencv.{modname}')
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and
                    hasattr(obj, 'get_name') and
                    hasattr(obj, 'draw') and
                    name != 'BaseEffect' and
                    name != 'BaseUIEffect' and
                    not name.startswith('Base')):
                    effects.append({
                        'module': modname,
                        'class_name': name,
                        'name': obj.get_name(),
                        'class': obj
                    })
        except Exception as e:
            pass

    # Sort by name
    effects.sort(key=lambda x: x['name'])
    return effects


class PipelineBuilderEffect(BaseUIEffect):
    """Dynamically build and chain multiple OpenCV effects together"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        self.width = width
        self.height = height
        self.root = root

        # List of effect instances in the pipeline
        self.effects = []

        # Available effects
        self.available_effects = get_opencv_effects()

        # UI references
        self.effects_container = None
        self.effect_frames = []

        # Pipeline name for saving
        self.pipeline_name = tk.StringVar(value="")

        # Add first button reference
        self.add_first_btn = None

        # Pipeline to load for editing (set by main.py from --edit-pipeline arg)
        self._pipeline_to_load = None

    @classmethod
    def get_name(cls) -> str:
        return "Pipeline Builder"

    @classmethod
    def get_description(cls) -> str:
        return "Build custom effect chains from OpenCV effects"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def create_control_panel(self, parent):
        """Create the pipeline builder UI"""
        self.control_panel = ttk.Frame(parent)

        padding = {'padx': 10, 'pady': 5}

        # Header section
        header_frame = ttk.Frame(self.control_panel)
        header_frame.pack(fill='x', **padding)

        # Title
        title_label = ttk.Label(
            header_frame,
            text="Pipeline Builder",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Pipeline name and save controls
        save_frame = ttk.Frame(self.control_panel)
        save_frame.pack(fill='x', **padding)

        ttk.Label(save_frame, text="Name:").pack(side='left')

        name_entry = ttk.Entry(
            save_frame,
            textvariable=self.pipeline_name,
            width=15
        )
        name_entry.pack(side='left', padx=5)

        save_btn = ttk.Button(
            save_frame,
            text="Save",
            command=self._save_pipeline
        )
        save_btn.pack(side='left', padx=5)

        load_btn = ttk.Button(
            save_frame,
            text="Load",
            command=self._show_load_dialog
        )
        load_btn.pack(side='left', padx=5)

        # Separator
        ttk.Separator(self.control_panel, orient='horizontal').pack(fill='x', pady=5)

        # Add first effect button
        add_first_frame = ttk.Frame(self.control_panel)
        add_first_frame.pack(fill='x', **padding)

        self.add_first_btn = ttk.Button(
            add_first_frame,
            text="+ Add First Effect",
            command=lambda: self._show_effect_selector(0)
        )
        self.add_first_btn.pack(side='left')

        # Container for effect panels
        self.effects_container = ttk.Frame(self.control_panel)
        self.effects_container.pack(fill='both', expand=True, **padding)

        # Load pipeline if one was specified for editing
        if self._pipeline_to_load:
            self._load_pipeline_by_key(self._pipeline_to_load)

        return self.control_panel

    def _show_effect_selector(self, insert_index):
        """Show dropdown to select an effect to add"""
        # Create a popup frame
        selector_frame = ttk.Frame(self.effects_container)
        selector_frame.pack(fill='x', pady=5)

        # Position it correctly
        if insert_index > 0 and insert_index <= len(self.effect_frames):
            selector_frame.pack_forget()
            if insert_index < len(self.effect_frames):
                selector_frame.pack(fill='x', pady=5, before=self.effect_frames[insert_index])
            else:
                selector_frame.pack(fill='x', pady=5)

        ttk.Label(selector_frame, text="Select Effect:").pack(side='left')

        # Effect dropdown
        effect_names = [e['name'] for e in self.available_effects]
        effect_var = tk.StringVar()

        combo = ttk.Combobox(
            selector_frame,
            textvariable=effect_var,
            values=effect_names,
            state='readonly',
            width=25
        )
        combo.pack(side='left', padx=5)

        def on_select(event=None):
            selected_name = effect_var.get()
            if selected_name:
                # Find the effect class
                for effect_info in self.available_effects:
                    if effect_info['name'] == selected_name:
                        self._add_effect(effect_info, insert_index)
                        break
            selector_frame.destroy()

        def on_cancel():
            selector_frame.destroy()

        combo.bind('<<ComboboxSelected>>', on_select)

        cancel_btn = ttk.Button(
            selector_frame,
            text="Cancel",
            command=on_cancel
        )
        cancel_btn.pack(side='left', padx=5)

        combo.focus_set()

    def _add_effect(self, effect_info, insert_index):
        """Add an effect to the pipeline at the specified index"""
        # Create effect instance
        effect_class = effect_info['class']
        effect = effect_class(self.width, self.height, self.root)

        # Insert into effects list
        self.effects.insert(insert_index, effect)

        # Create UI for this effect
        effect_frame = self._create_effect_ui(effect, effect_info['name'], insert_index)

        # Insert frame at correct position
        self.effect_frames.insert(insert_index, effect_frame)

        # Repack all frames in order
        self._repack_effect_frames()

        # Update indices for all effect UIs
        self._update_effect_indices()

        # Hide "add first" button if we have effects
        if self.effects and self.add_first_btn:
            self.add_first_btn.pack_forget()

    def _create_effect_ui(self, effect, effect_name, index):
        """Create the UI panel for a single effect in the pipeline"""
        # Main frame for this effect
        effect_frame = ttk.LabelFrame(
            self.effects_container,
            text=f"{effect_name}"
        )

        # Store index as attribute
        effect_frame.effect_index = index

        # Main container with two columns
        main_container = ttk.Frame(effect_frame)
        main_container.pack(fill='both', expand=True, padx=5, pady=2)

        # Left column - Enabled checkbox and +/- buttons
        left_column = ttk.Frame(main_container)
        left_column.pack(side='left', anchor='n', padx=(0, 10), pady=5)

        # Enabled checkbox (using effect's variable)
        if hasattr(effect, 'enabled'):
            enabled_cb = ttk.Checkbutton(
                left_column,
                text="Enabled",
                variable=effect.enabled
            )
            enabled_cb.pack(anchor='w')

        # +/- buttons in a row below Enabled
        btn_frame = ttk.Frame(left_column)
        btn_frame.pack(anchor='w', pady=(2, 0))

        plus_btn = ttk.Button(
            btn_frame,
            text="+",
            width=2,
            command=lambda f=effect_frame: self._show_effect_selector(self._get_frame_index(f) + 1)
        )
        plus_btn.pack(side='left', padx=(0, 2))

        minus_btn = ttk.Button(
            btn_frame,
            text="-",
            width=2,
            command=lambda f=effect_frame: self._remove_effect(f)
        )
        minus_btn.pack(side='left')

        # Right column - Effect's own control panel
        right_column = ttk.Frame(main_container)
        right_column.pack(side='left', fill='both', expand=True)

        if hasattr(effect, 'create_control_panel'):
            effect_panel = effect.create_control_panel(right_column)
            if effect_panel:
                effect_panel.pack(fill='x')

        return effect_frame

    def _get_frame_index(self, frame):
        """Get the current index of a frame"""
        try:
            return self.effect_frames.index(frame)
        except ValueError:
            return 0

    def _remove_effect(self, effect_frame):
        """Remove an effect from the pipeline"""
        index = self._get_frame_index(effect_frame)

        if index < len(self.effects):
            # Cleanup the effect
            effect = self.effects[index]
            if hasattr(effect, 'cleanup'):
                effect.cleanup()

            # Remove from lists
            self.effects.pop(index)
            self.effect_frames.pop(index)

            # Destroy UI
            effect_frame.destroy()

            # Update indices
            self._update_effect_indices()

            # Show "add first" button if no effects
            if not self.effects and self.add_first_btn:
                self.add_first_btn.pack(side='left')

    def _repack_effect_frames(self):
        """Repack all effect frames in order"""
        for frame in self.effect_frames:
            frame.pack_forget()
        for frame in self.effect_frames:
            frame.pack(fill='x', pady=5)

    def _update_effect_indices(self):
        """Update the stored index in each effect frame"""
        for i, frame in enumerate(self.effect_frames):
            frame.effect_index = i

    def _save_pipeline(self):
        """Save the current pipeline configuration"""
        name = self.pipeline_name.get().strip()
        if not name:
            messagebox.showwarning("Save Pipeline", "Please enter a pipeline name")
            return

        if not self.effects:
            messagebox.showwarning("Save Pipeline", "Pipeline is empty - add some effects first")
            return

        # Build pipeline config
        config = {
            'name': name,
            'effects': []
        }

        for i, effect in enumerate(self.effects):
            effect_config = {
                'module': None,
                'class_name': effect.__class__.__name__,
                'params': {}
            }

            # Find module name
            for info in self.available_effects:
                if info['class'] == effect.__class__:
                    effect_config['module'] = info['module']
                    break

            # Save parameter values
            for attr_name in dir(effect):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(effect, attr_name)
                if isinstance(attr, tk.Variable):
                    try:
                        effect_config['params'][attr_name] = attr.get()
                    except:
                        pass

            config['effects'].append(effect_config)

        # Load existing pipelines
        pipelines_file = os.path.expanduser('~/.webcam_filters_pipelines.json')
        pipelines = {}
        if os.path.exists(pipelines_file):
            try:
                with open(pipelines_file, 'r') as f:
                    pipelines = json.load(f)
            except:
                pass

        # Save this pipeline
        pipeline_key = f"user_{name}"
        pipelines[pipeline_key] = config

        with open(pipelines_file, 'w') as f:
            json.dump(pipelines, f, indent=2)

        messagebox.showinfo("Save Pipeline", f"Pipeline '{pipeline_key}' saved successfully!")

    def _show_load_dialog(self):
        """Show dialog to load a saved pipeline"""
        pipelines_file = os.path.expanduser('~/.webcam_filters_pipelines.json')

        if not os.path.exists(pipelines_file):
            messagebox.showinfo("Load Pipeline", "No saved pipelines found")
            return

        try:
            with open(pipelines_file, 'r') as f:
                pipelines = json.load(f)
        except:
            messagebox.showerror("Load Pipeline", "Error reading pipelines file")
            return

        if not pipelines:
            messagebox.showinfo("Load Pipeline", "No saved pipelines found")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Pipeline")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select Pipeline:").pack(pady=10)

        pipeline_var = tk.StringVar()
        pipeline_names = list(pipelines.keys())

        combo = ttk.Combobox(
            dialog,
            textvariable=pipeline_var,
            values=pipeline_names,
            state='readonly',
            width=30
        )
        combo.pack(pady=5)

        def on_load():
            name = pipeline_var.get()
            if name:
                self._load_pipeline(pipelines[name])
                self.pipeline_name.set(name.replace('user_', ''))
            dialog.destroy()

        def on_delete():
            name = pipeline_var.get()
            if name:
                if messagebox.askyesno("Delete Pipeline", f"Delete '{name}'?"):
                    del pipelines[name]
                    with open(pipelines_file, 'w') as f:
                        json.dump(pipelines, f, indent=2)
                    combo['values'] = list(pipelines.keys())
                    pipeline_var.set('')

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Load", command=on_load).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Delete", command=on_delete).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)

    def _load_pipeline_by_key(self, pipeline_key):
        """Load a pipeline by its key (e.g., 'user_test1')"""
        pipelines_file = os.path.expanduser('~/.webcam_filters_pipelines.json')

        if not os.path.exists(pipelines_file):
            print(f"Warning: No pipelines file found")
            return

        try:
            with open(pipelines_file, 'r') as f:
                pipelines = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load pipelines: {e}")
            return

        if pipeline_key not in pipelines:
            print(f"Warning: Pipeline '{pipeline_key}' not found")
            return

        config = pipelines[pipeline_key]
        self._load_pipeline(config)
        # Set the name (strip 'user_' prefix)
        name = pipeline_key.replace('user_', '')
        self.pipeline_name.set(name)

    def _load_pipeline(self, config):
        """Load a pipeline from configuration"""
        # Clear existing effects
        for frame in self.effect_frames[:]:
            self._remove_effect(frame)

        # Load effects
        for effect_config in config['effects']:
            # Find effect info
            for info in self.available_effects:
                if info['module'] == effect_config['module'] and \
                   info['class_name'] == effect_config['class_name']:
                    # Add the effect
                    self._add_effect(info, len(self.effects))

                    # Restore parameters
                    effect = self.effects[-1]
                    for param_name, value in effect_config['params'].items():
                        if hasattr(effect, param_name):
                            attr = getattr(effect, param_name)
                            if isinstance(attr, tk.Variable):
                                try:
                                    attr.set(value)
                                except:
                                    pass
                    break

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply all effects in the pipeline"""
        result = frame

        for effect in self.effects:
            # Check if effect is enabled
            if hasattr(effect, 'enabled'):
                if not effect.enabled.get():
                    continue

            # Update if needed
            if hasattr(effect, 'update'):
                effect.update()

            # Apply the effect
            result = effect.draw(result, face_mask)

        return result

    def cleanup(self):
        """Cleanup all effects in the pipeline"""
        for effect in self.effects:
            if hasattr(effect, 'cleanup'):
                effect.cleanup()
