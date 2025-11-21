"""
Pipeline Builder 2 - FormRenderer-enabled pipeline builder.

Uses the new FormRenderer system for effects that support it.
Provides UI to add/remove/reorder OpenCV effects at runtime.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from core.base_effect import BaseUIEffect


def _create_tooltip(widget, text):
    """Create a tooltip for a widget"""
    def show_tooltip(event):
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        label = ttk.Label(tooltip, text=text, background="#ffffe0",
                         relief='solid', borderwidth=1, padding=(5, 2))
        label.pack()
        widget._tooltip = tooltip

    def hide_tooltip(event):
        if hasattr(widget, '_tooltip'):
            widget._tooltip.destroy()
            del widget._tooltip

    widget.bind('<Enter>', show_tooltip)
    widget.bind('<Leave>', hide_tooltip)


def _extract_tk_variables(obj, visited=None):
    """Recursively extract tk.Variable values from an object.

    Handles:
    - Direct tk.Variable attributes
    - Dicts containing tk.Variables or nested dicts/lists
    - Lists containing tk.Variables or nested dicts/lists

    Returns a serializable structure with the same shape.
    """
    if visited is None:
        visited = set()

    # Avoid infinite recursion
    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    if isinstance(obj, tk.Variable):
        try:
            return obj.get()
        except:
            return None
    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            extracted = _extract_tk_variables(value, visited)
            if extracted is not None:
                result[key] = extracted
        return result if result else None
    elif isinstance(obj, list):
        result = []
        has_values = False
        for item in obj:
            extracted = _extract_tk_variables(item, visited)
            result.append(extracted)
            if extracted is not None:
                has_values = True
        return result if has_values else None
    else:
        return None


def _restore_tk_variables(obj, data):
    """Recursively restore tk.Variable values from serialized data.

    Handles:
    - Direct tk.Variable attributes
    - Dicts containing tk.Variables or nested dicts/lists
    - Lists containing tk.Variables or nested dicts/lists
    """
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


def get_opencv_effects():
    """Get list of available OpenCV effects (excluding pipeline classes)"""
    import importlib
    import pkgutil

    effects = []
    effects_dir = os.path.dirname(__file__)

    for importer, modname, ispkg in pkgutil.iter_modules([effects_dir]):
        # Skip pipeline-related modules
        if modname in ('pipeline', 'pipeline_builder', 'pipeline_builder2'):
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


class PipelineBuilder2Effect(BaseUIEffect):
    """Dynamically build and chain multiple OpenCV effects together (FormRenderer version)"""

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
        self.current_selector = None

        # Pipeline name and description for saving
        self.pipeline_name = tk.StringVar(value="")
        self.pipeline_description = tk.StringVar(value="")
        self.all_enabled = tk.BooleanVar(value=True)

        # Add first button/frame references
        self.add_first_btn = None
        self.add_first_frame = None

        # Pipeline to load for editing (set by main.py from --edit-pipeline arg)
        self._pipeline_to_load = None

    @classmethod
    def get_name(cls) -> str:
        return "Pipeline Builder 2"

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

        # Warning message
        warning_label = ttk.Label(
            header_frame,
            text="⚠️  Changes are not persisted until you click Save",
            foreground='red',
            font=('TkDefaultFont', 10, 'italic')
        )
        warning_label.pack(anchor='w', pady=(5, 0))

        # Pipeline name/description and save controls using grid for alignment
        fields_frame = ttk.Frame(self.control_panel)
        fields_frame.pack(fill='x', **padding)

        # Name row
        ttk.Label(fields_frame, text="Name:").grid(row=0, column=0, sticky='e', padx=(0, 5), pady=2)

        name_entry = ttk.Entry(
            fields_frame,
            textvariable=self.pipeline_name,
            width=15
        )
        name_entry.grid(row=0, column=1, sticky='w', pady=2)

        # Description row
        ttk.Label(fields_frame, text="Description:").grid(row=1, column=0, sticky='e', padx=(0, 5), pady=2)

        desc_entry = ttk.Entry(
            fields_frame,
            textvariable=self.pipeline_description,
            width=40
        )
        desc_entry.grid(row=1, column=1, sticky='ew', pady=2)

        # All checkbox in column 0
        all_frame = ttk.Frame(fields_frame)
        all_frame.grid(row=2, column=0, pady=(5, 2))

        ttk.Label(all_frame, text="All:").pack(side='left')
        ttk.Checkbutton(all_frame, variable=self.all_enabled).pack(side='left')

        # Buttons row
        btn_frame = ttk.Frame(fields_frame)
        btn_frame.grid(row=2, column=1, sticky='e', pady=(5, 2))

        cancel_btn = tk.Label(
            btn_frame, text="Cancel", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        cancel_btn.pack(side='left', padx=(0, 5))
        cancel_btn.bind('<Button-1>', lambda e: self._run_pipeline())
        _create_tooltip(cancel_btn, "Exit pipeline builder")

        save_btn = tk.Label(
            btn_frame, text="Save All", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        save_btn.pack(side='left')
        save_btn.bind('<Button-1>', lambda e: self._save_pipeline())
        _create_tooltip(save_btn, "Save pipeline to file")

        # Clipboard buttons (styled like effect buttons)
        ct_btn = tk.Label(
            btn_frame, text="CT", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        ct_btn.pack(side='left', padx=(10, 1))
        ct_btn.bind('<Button-1>', lambda e: self._copy_text())
        _create_tooltip(ct_btn, "Copy as text")

        pt_btn = tk.Label(
            btn_frame, text="PT", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        pt_btn.pack(side='left', padx=1)
        _create_tooltip(pt_btn, "Paste from text")

        cj_btn = tk.Label(
            btn_frame, text="CJ", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        cj_btn.pack(side='left', padx=1)
        cj_btn.bind('<Button-1>', lambda e: self._copy_json())
        _create_tooltip(cj_btn, "Copy as JSON")

        pj_btn = tk.Label(
            btn_frame, text="PJ", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        pj_btn.pack(side='left', padx=1)
        pj_btn.bind('<Button-1>', lambda e: self._paste_json())
        _create_tooltip(pj_btn, "Paste from JSON")

        # Configure column weights
        fields_frame.columnconfigure(1, weight=1)

        # Separator
        ttk.Separator(self.control_panel, orient='horizontal').pack(fill='x', pady=5)

        # Add first effect button (in its own frame so we can destroy it completely)
        self.add_first_frame = ttk.Frame(self.control_panel)
        self.add_first_frame.pack(fill='x', **padding)

        self.add_first_btn = tk.Label(
            self.add_first_frame,
            text="+ Add First Effect",
            relief='raised',
            borderwidth=1,
            padx=2,
            pady=0,
            cursor='arrow'
        )
        self.add_first_btn.pack(side='left')
        self.add_first_btn.bind('<Button-1>', lambda e: self._show_effect_selector(0))

        # Container for effect panels (minimal padding to reduce whitespace)
        self.effects_container = ttk.Frame(self.control_panel)
        self.effects_container.pack(fill='both', expand=True, padx=10, pady=0)

        # Load pipeline if one was specified for editing
        if self._pipeline_to_load:
            self._load_pipeline_by_key(self._pipeline_to_load)

        return self.control_panel

    def _show_effect_selector(self, insert_index):
        """Show dropdown to select an effect to add"""
        # Close any existing selector
        if self.current_selector is not None:
            try:
                self.current_selector.destroy()
            except:
                pass
            self.current_selector = None

        # Create a popup frame
        selector_frame = ttk.Frame(self.effects_container)
        selector_frame.pack(fill='x', pady=5)
        self.current_selector = selector_frame

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
        print(f"Dropdown effects ({len(effect_names)}): {effect_names[:10]}...")
        effect_var = tk.StringVar()

        combo = ttk.Combobox(
            selector_frame,
            textvariable=effect_var,
            values=effect_names,
            state='readonly',
            width=25,
            height=20
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
            self.current_selector = None

        def on_cancel():
            selector_frame.destroy()
            self.current_selector = None

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

        # Hide "add first" frame if we have effects
        if self.effects and self.add_first_frame:
            self.add_first_frame.pack_forget()

    def _create_effect_ui(self, effect, effect_name, index):
        """Create the UI panel for a single effect in the pipeline"""
        # Check if effect uses new FormRenderer system
        uses_form_renderer = hasattr(effect, 'get_form_schema')

        if uses_form_renderer:
            # New FormRenderer effects render their own container with name/desc/sig
            effect_frame = ttk.Frame(self.effects_container)
            effect_frame.effect_index = index

            # Set pipeline callbacks for +/- buttons
            effect._on_add_below = lambda f=effect_frame: self._show_effect_selector(self._get_frame_index(f) + 1)
            effect._on_remove = lambda f=effect_frame: self._remove_effect(f)

            # Set flag and create control panel in view mode
            effect._in_pipeline = True
            effect_panel = effect.create_control_panel(effect_frame, mode='view')
            if effect_panel:
                effect_panel.pack(fill='x', padx=5, pady=2)
        else:
            # Old-style effects need the LabelFrame wrapper
            effect_frame = ttk.LabelFrame(
                self.effects_container,
                text=effect_name
            )
            effect_frame.effect_index = index

            # Description on its own line, inside frame at top
            if hasattr(effect, 'get_description'):
                desc = effect.get_description()
                if desc:
                    ttk.Label(
                        effect_frame,
                        text=desc,
                        font=('TkDefaultFont', 10)
                    ).pack(anchor='w', padx=10, pady=(5, 0))

            # Method signature on its own line, below description
            if hasattr(effect, 'get_method_signature'):
                sig = effect.get_method_signature()
                if sig:
                    ttk.Label(
                        effect_frame,
                        text=sig,
                        font=('TkFixedFont', 10)
                    ).pack(anchor='w', padx=10, pady=(2, 0))

            # Effect's own control panel
            if hasattr(effect, 'create_control_panel'):
                effect._in_pipeline = True
                effect_panel = effect.create_control_panel(effect_frame)
            if effect_panel:
                effect_panel.pack(fill='x', padx=5, pady=2)

                # Find the left_column in the effect's panel and add +/- buttons there
                # The left_column contains the Enabled checkbox
                self._add_buttons_to_left_column(effect_panel, effect_frame)

            # Show any visualization windows created by the effect (e.g., FFT)
            if hasattr(effect, 'viz_window') and effect.viz_window is not None:
                effect.viz_window.deiconify()
            if hasattr(effect, 'diff_window') and effect.diff_window is not None:
                effect.diff_window.deiconify()

        return effect_frame

    def _add_buttons_to_left_column(self, effect_panel, effect_frame):
        """Find the left column in the effect panel and add +/- buttons below Enabled"""
        # Search for the left_column frame that contains the Enabled checkbox
        # It's typically in a main_frame which is a child of control_panel
        for child in effect_panel.winfo_children():
            if isinstance(child, ttk.Frame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, ttk.Frame):
                        # Check if this frame contains a Checkbutton (Enabled)
                        for widget in subchild.winfo_children():
                            if isinstance(widget, ttk.Checkbutton):
                                # Found the left column - add buttons here
                                btn_frame = ttk.Frame(subchild)
                                btn_frame.pack(pady=(2, 0))

                                plus_btn = ttk.Button(
                                    btn_frame,
                                    text="+",
                                    width=1,
                                    command=lambda f=effect_frame: self._show_effect_selector(self._get_frame_index(f) + 1)
                                )
                                plus_btn.pack(side='left', padx=(0, 1))
                                _create_tooltip(plus_btn, "Add effect below")

                                minus_btn = ttk.Button(
                                    btn_frame,
                                    text="-",
                                    width=1,
                                    command=lambda f=effect_frame: self._remove_effect(f)
                                )
                                minus_btn.pack(side='left')
                                _create_tooltip(minus_btn, "Remove this effect")
                                return

        # Fallback: add buttons at the bottom if we couldn't find left column
        btn_frame = ttk.Frame(effect_frame)
        btn_frame.pack(anchor='w', padx=10, pady=(0, 5))

        plus_btn = ttk.Button(
            btn_frame,
            text="+",
            width=1,
            command=lambda f=effect_frame: self._show_effect_selector(self._get_frame_index(f) + 1)
        )
        plus_btn.pack(side='left', padx=(0, 1))
        _create_tooltip(plus_btn, "Add effect below")

        minus_btn = ttk.Button(
            btn_frame,
            text="-",
            width=1,
            command=lambda f=effect_frame: self._remove_effect(f)
        )
        minus_btn.pack(side='left')
        _create_tooltip(minus_btn, "Remove this effect")

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

            # Show "add first" frame if no effects
            if not self.effects and self.add_first_frame:
                self.add_first_frame.pack(fill='x', padx=10, pady=5)

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
            'description': self.pipeline_description.get().strip(),
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

            # Save parameter values - recursively extract all tk.Variables
            # including those in nested dicts and lists
            for attr_name in dir(effect):
                if attr_name.startswith('_'):
                    continue
                # Skip methods and non-data attributes
                try:
                    attr = getattr(effect, attr_name)
                except:
                    continue

                # Handle direct tk.Variable
                if isinstance(attr, tk.Variable):
                    try:
                        effect_config['params'][attr_name] = attr.get()
                    except:
                        pass
                # Handle dicts and lists that may contain tk.Variables
                elif isinstance(attr, (dict, list)):
                    extracted = _extract_tk_variables(attr)
                    if extracted is not None:
                        effect_config['params'][attr_name] = extracted

            config['effects'].append(effect_config)

        # Get pipelines directory (in project root)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pipelines_dir = os.path.join(project_root, 'pipelines')

        # Create directory if it doesn't exist
        os.makedirs(pipelines_dir, exist_ok=True)

        # Save this pipeline as its own file
        pipeline_file = os.path.join(pipelines_dir, f"{name}.json")

        with open(pipeline_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Pipeline saved: '{name}' -> {pipeline_file}")

        # Generate event to notify main.py to refresh the effect list
        if self.root_window:
            self.root_window.event_generate('<<PipelineSaved>>', when='tail')

    def _run_pipeline(self):
        """Switch to running the current pipeline (exit edit mode)"""
        name = self.pipeline_name.get().strip()
        if not name:
            messagebox.showwarning("Run Pipeline", "Please enter a pipeline name and save first")
            return

        # Check if pipeline exists
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pipelines_dir = os.path.join(project_root, 'pipelines')
        pipeline_file = os.path.join(pipelines_dir, f"{name}.json")

        if not os.path.exists(pipeline_file):
            messagebox.showwarning("Run Pipeline", "Please save the pipeline first")
            return

        # Generate event to switch to the user pipeline
        if self.root_window:
            self.root_window.event_generate('<<RunPipeline>>')

    def _load_pipeline_by_key(self, pipeline_key):
        """Load a pipeline by its key (e.g., 'test1')"""
        # Get pipelines directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pipelines_dir = os.path.join(project_root, 'pipelines')

        # Handle old-style keys with 'user_' prefix
        name = pipeline_key.replace('user_', '')
        pipeline_file = os.path.join(pipelines_dir, f"{name}.json")

        if not os.path.exists(pipeline_file):
            print(f"Warning: Pipeline file not found: {pipeline_file}")
            return

        try:
            with open(pipeline_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load pipeline: {e}")
            return

        self._load_pipeline(config)
        self.pipeline_name.set(name)
        self.pipeline_description.set(config.get('description', ''))
        print(f"Pipeline loaded: '{name}' <- {pipeline_file}")

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

                    # Restore parameters - recursively restore tk.Variables
                    # including those in nested dicts and lists
                    effect = self.effects[-1]
                    for param_name, value in effect_config['params'].items():
                        if hasattr(effect, param_name):
                            attr = getattr(effect, param_name)
                            if isinstance(attr, tk.Variable):
                                try:
                                    attr.set(value)
                                except:
                                    pass
                            elif isinstance(attr, (dict, list)):
                                _restore_tk_variables(attr, value)
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

    def _copy_text(self):
        """Copy pipeline settings as human-readable text to clipboard"""
        lines = []
        lines.append(f"Pipeline: {self.pipeline_name.get()}")
        if self.pipeline_description.get():
            lines.append(f"Description: {self.pipeline_description.get()}")
        lines.append("")

        for i, effect in enumerate(self.effects):
            effect_name = effect.get_name() if hasattr(effect, 'get_name') else effect.__class__.__name__
            lines.append(f"{i+1}. {effect_name}")
            if hasattr(effect, 'get_view_mode_summary'):
                summary = effect.get_view_mode_summary()
                for line in summary.split('\n'):
                    lines.append(f"   {line}")
            lines.append("")

        text = '\n'.join(lines)
        if self.root:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def _copy_json(self):
        """Copy pipeline as JSON to clipboard - same format as save file"""
        name = self.pipeline_name.get().strip()
        config = {
            'name': name,
            'description': self.pipeline_description.get().strip(),
            'effects': []
        }

        for effect in self.effects:
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
                try:
                    attr = getattr(effect, attr_name)
                except:
                    continue

                if isinstance(attr, tk.Variable):
                    try:
                        effect_config['params'][attr_name] = attr.get()
                    except:
                        pass
                elif isinstance(attr, (dict, list)):
                    extracted = _extract_tk_variables(attr)
                    if extracted is not None:
                        effect_config['params'][attr_name] = extracted

            config['effects'].append(effect_config)

        text = json.dumps(config, indent=2)
        if self.root:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def _paste_text(self):
        """Paste pipeline from text - not implemented for pipeline level"""
        pass

    def _paste_json(self):
        """Paste pipeline from JSON on clipboard"""
        if not self.root:
            return

        try:
            text = self.root.clipboard_get()
            config = json.loads(text)

            # Load the pipeline
            self._load_pipeline(config)
            self.pipeline_name.set(config.get('name', ''))
            self.pipeline_description.set(config.get('description', ''))
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def cleanup(self):
        """Cleanup all effects in the pipeline"""
        for effect in self.effects:
            if hasattr(effect, 'cleanup'):
                effect.cleanup()
