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


# Constants
TEXT_SEPARATOR_LENGTH = 40


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

        # View/edit mode for the pipeline
        self._current_mode = 'view'
        self._control_parent = None

        # UI references for view/edit mode switching
        self._warning_label = None
        self._name_entry = None
        self._name_label = None
        self._desc_entry = None
        self._desc_label = None
        self._fields_frame = None

        # Store original values for cancel functionality
        self._original_name = ""
        self._original_description = ""

    def _ensure_edit_mode(self):
        """Switch to edit mode if not already in it - called when user takes any editing action"""
        if self._current_mode != 'edit':
            # Enter edit mode - store current values
            self._original_name = self.pipeline_name.get()
            self._original_description = self.pipeline_description.get()
            self._current_mode = 'edit'
            self._update_mode_ui()

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
        self._control_parent = parent
        self.control_panel = ttk.Frame(parent)

        padding = {'padx': 10, 'pady': 5}

        # Header section - use grid for title and warning on same row
        header_frame = ttk.Frame(self.control_panel)
        header_frame.pack(fill='x', **padding)

        # Title on left
        title_label = ttk.Label(
            header_frame,
            text="Pipeline Builder",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.grid(row=0, column=0, sticky='w')

        # Warning message on right (only shown in edit mode)
        self._warning_label = ttk.Label(
            header_frame,
            text="⚠️  Changes are not persisted until you click Save All",
            foreground='red',
            font=('TkDefaultFont', 10, 'italic')
        )
        if self._current_mode == 'edit':
            self._warning_label.grid(row=0, column=1, sticky='e')

        # Configure column weights so warning is right-justified
        header_frame.columnconfigure(1, weight=1)

        # Pipeline name/description and save controls using grid for alignment
        self._fields_frame = ttk.Frame(self.control_panel)
        self._fields_frame.pack(fill='x', **padding)

        # Name row
        ttk.Label(self._fields_frame, text="Name:").grid(row=0, column=0, sticky='e', padx=(0, 5), pady=4)

        # Name entry (edit mode)
        self._name_entry = ttk.Entry(
            self._fields_frame,
            textvariable=self.pipeline_name,
            width=15
        )

        # Name label (view mode)
        self._name_label = ttk.Label(
            self._fields_frame,
            textvariable=self.pipeline_name
        )

        # Show appropriate widget based on mode
        if self._current_mode == 'edit':
            self._name_entry.grid(row=0, column=1, sticky='w', pady=0)
        else:
            self._name_label.grid(row=0, column=1, sticky='w', pady=0)

        # Description row
        ttk.Label(self._fields_frame, text="Description:").grid(row=1, column=0, sticky='e', padx=(0, 5), pady=4)

        # Description entry (edit mode)
        self._desc_entry = ttk.Entry(
            self._fields_frame,
            textvariable=self.pipeline_description,
            width=40
        )

        # Description label (view mode)
        self._desc_label = ttk.Label(
            self._fields_frame,
            textvariable=self.pipeline_description
        )

        # Show appropriate widget based on mode
        if self._current_mode == 'edit':
            self._desc_entry.grid(row=1, column=1, sticky='ew', pady=0)
        else:
            self._desc_label.grid(row=1, column=1, sticky='w', pady=0)

        # All checkbox in column 0
        all_frame = ttk.Frame(self._fields_frame)
        all_frame.grid(row=2, column=0, pady=(5, 2))

        ttk.Label(all_frame, text="All:").pack(side='left')
        ttk.Checkbutton(all_frame, variable=self.all_enabled).pack(side='left')

        # Wire up the All checkbox to toggle all effects
        self.all_enabled.trace_add('write', lambda *args: self._toggle_all_effects())

        # Buttons row
        btn_frame = ttk.Frame(self._fields_frame)
        btn_frame.grid(row=2, column=1, sticky='e', pady=(5, 2))

        # Cancel All button (only shown in edit mode)
        self._cancel_btn = tk.Label(
            btn_frame, text="Cancel All", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        self._cancel_btn.bind('<Button-1>', lambda e: self._toggle_pipeline_mode())
        _create_tooltip(self._cancel_btn, "Switch to view mode")

        # Edit All / Save All button
        self._edit_save_btn = tk.Label(
            btn_frame, text="Edit All", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        self._edit_save_btn.bind('<Button-1>', lambda e: self._on_edit_save_click())
        _create_tooltip(self._edit_save_btn, "Edit all effects")

        # Clipboard buttons (styled like effect buttons)
        self._ct_btn = tk.Label(
            btn_frame, text="CT", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        self._ct_btn.bind('<Button-1>', lambda e: self._copy_text())
        _create_tooltip(self._ct_btn, "Copy as text")

        self._pt_btn = tk.Label(
            btn_frame, text="PT", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        self._pt_btn.bind('<Button-1>', lambda e: self._paste_text() if self._current_mode == 'edit' else None)
        _create_tooltip(self._pt_btn, "Paste from text")

        self._cj_btn = tk.Label(
            btn_frame, text="CJ", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        self._cj_btn.bind('<Button-1>', lambda e: self._copy_json())
        _create_tooltip(self._cj_btn, "Copy as JSON")

        self._pj_btn = tk.Label(
            btn_frame, text="PJ", relief='raised', borderwidth=1,
            padx=2, pady=0, cursor='arrow'
        )
        self._pj_btn.bind('<Button-1>', lambda e: self._paste_json() if self._current_mode == 'edit' else None)
        _create_tooltip(self._pj_btn, "Paste from JSON")

        # Gray out paste buttons in view mode (no fg set in edit mode - use system default)
        if self._current_mode == 'view':
            self._pt_btn.config(fg='gray')
            self._pj_btn.config(fg='gray')

        # Pack all buttons in correct order based on mode
        if self._current_mode == 'edit':
            self._cancel_btn.pack(side='left', padx=(0, 5))
            self._edit_save_btn.pack(side='left')
            self._edit_save_btn.config(text="Save All")
        else:
            self._edit_save_btn.pack(side='left')
            self._edit_save_btn.config(text="Edit All")

        self._ct_btn.pack(side='left', padx=(10, 1))
        self._pt_btn.pack(side='left', padx=1)
        self._cj_btn.pack(side='left', padx=1)
        self._pj_btn.pack(side='left', padx=1)

        # Configure column weights
        self._fields_frame.columnconfigure(1, weight=1)

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
        # Ensure we're in edit mode when adding effects
        self._ensure_edit_mode()

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

        # Add trace on effect's enabled variable to trigger edit mode
        if hasattr(effect, 'enabled'):
            effect.enabled.trace_add('write', lambda *args: self._ensure_edit_mode())

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

            # Set pipeline callbacks for +/- buttons and edit mode notification
            effect._on_add_below = lambda f=effect_frame: self._show_effect_selector(self._get_frame_index(f) + 1)
            effect._on_remove = lambda f=effect_frame: self._remove_effect(f)
            effect._on_edit = self._ensure_edit_mode  # Notify pipeline when effect enters edit mode

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
        # Ensure we're in edit mode when removing effects
        self._ensure_edit_mode()

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

                    # Re-render the effect's control panel to show updated values
                    if hasattr(effect, '_toggle_mode'):
                        # Toggle twice to stay in same mode but refresh display
                        effect._toggle_mode()
                        effect._toggle_mode()

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

    def _toggle_all_effects(self):
        """Toggle all effects' enabled state based on the All checkbox"""
        # Ensure we're in edit mode when toggling enabled state
        self._ensure_edit_mode()

        enabled = self.all_enabled.get()
        for effect in self.effects:
            if hasattr(effect, 'enabled'):
                effect.enabled.set(enabled)

    def _toggle_pipeline_mode(self):
        """Toggle between view and edit modes (Cancel All button)"""
        if self._current_mode == 'edit':
            # Cancel - restore original values
            self.pipeline_name.set(self._original_name)
            self.pipeline_description.set(self._original_description)
            self._current_mode = 'view'
        else:
            # Enter edit mode - store current values
            self._original_name = self.pipeline_name.get()
            self._original_description = self.pipeline_description.get()
            self._current_mode = 'edit'

        self._update_mode_ui()

    def _on_edit_save_click(self):
        """Handle Edit All / Save All button click"""
        if self._current_mode == 'view':
            # Enter edit mode - store current values
            self._original_name = self.pipeline_name.get()
            self._original_description = self.pipeline_description.get()
            self._current_mode = 'edit'
        else:
            # Save and switch to view mode
            self._save_pipeline()
            self._current_mode = 'view'

        self._update_mode_ui()

    def _update_mode_ui(self):
        """Update UI elements based on current mode"""
        # Update warning label visibility
        if self._current_mode == 'edit':
            self._warning_label.grid(row=0, column=1, sticky='e')
        else:
            self._warning_label.grid_forget()

        # Update name field (entry vs label)
        if self._current_mode == 'edit':
            self._name_label.grid_forget()
            self._name_entry.grid(row=0, column=1, sticky='w', pady=0)
        else:
            self._name_entry.grid_forget()
            self._name_label.grid(row=0, column=1, sticky='w', pady=0)

        # Update description field (entry vs label)
        if self._current_mode == 'edit':
            self._desc_label.grid_forget()
            self._desc_entry.grid(row=1, column=1, sticky='ew', pady=0)
        else:
            self._desc_entry.grid_forget()
            self._desc_label.grid(row=1, column=1, sticky='w', pady=0)

        # Update Cancel All button visibility and Edit/Save button text
        # Need to repack all buttons in correct order
        self._cancel_btn.pack_forget()
        self._edit_save_btn.pack_forget()
        self._ct_btn.pack_forget()
        self._pt_btn.pack_forget()
        self._cj_btn.pack_forget()
        self._pj_btn.pack_forget()

        if self._current_mode == 'edit':
            self._cancel_btn.pack(side='left', padx=(0, 5))
            self._edit_save_btn.pack(side='left')
            self._edit_save_btn.config(text="Save All")
        else:
            self._edit_save_btn.pack(side='left')
            self._edit_save_btn.config(text="Edit All")

        self._ct_btn.pack(side='left', padx=(10, 1))
        self._pt_btn.pack(side='left', padx=1)
        self._cj_btn.pack(side='left', padx=1)
        self._pj_btn.pack(side='left', padx=1)

        # Update paste button colors based on mode
        if self._current_mode == 'edit':
            # Reset to system default by getting the default from another label
            default_fg = self._ct_btn.cget('foreground')
            if default_fg == 'gray':
                # If CT is also gray, get the original default
                default_fg = 'SystemButtonText'
            self._pt_btn.config(fg=default_fg)
            self._pj_btn.config(fg=default_fg)
        else:
            self._pt_btn.config(fg='gray')
            self._pj_btn.config(fg='gray')

        # Update Add First Effect button visibility
        if self.add_first_frame:
            if self._current_mode == 'edit' and not self.effects:
                self.add_first_frame.pack(fill='x', padx=10, pady=5)
            elif self._current_mode == 'view':
                self.add_first_frame.pack_forget()

        # Update all sub-effects to match pipeline mode
        for effect in self.effects:
            if hasattr(effect, '_toggle_mode') and hasattr(effect, '_current_mode'):
                # Only toggle if sub-effect mode doesn't match pipeline mode
                if effect._current_mode != self._current_mode:
                    effect._toggle_mode()

    def _copy_text(self):
        """Copy pipeline settings as human-readable text to clipboard"""
        lines = []
        lines.append(f"Pipeline: {self.pipeline_name.get()}")
        if self.pipeline_description.get():
            lines.append(f"Description: {self.pipeline_description.get()}")

        separator = '-' * TEXT_SEPARATOR_LENGTH

        for effect in self.effects:
            lines.append(separator)
            # Get effect's text output (same format as effect's _copy_text)
            effect_name = effect.get_name() if hasattr(effect, 'get_name') else effect.__class__.__name__
            lines.append(effect_name)
            if hasattr(effect, 'get_description'):
                lines.append(effect.get_description())
            if hasattr(effect, 'get_method_signature'):
                lines.append(effect.get_method_signature())
            if hasattr(effect, 'get_view_mode_summary'):
                summary = effect.get_view_mode_summary()
                lines.append(summary)

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
        """Paste pipeline from human-readable text on clipboard"""
        if not self.root:
            return

        try:
            text = self.root.clipboard_get()
            lines = text.strip().split('\n')

            if not lines:
                return

            # Parse header (Pipeline name and description)
            pipeline_name = ""
            pipeline_description = ""
            separator = '-' * TEXT_SEPARATOR_LENGTH

            # Find pipeline name and description before first separator
            header_lines = []
            effect_sections = []
            current_section = []
            found_first_separator = False

            for line in lines:
                if line == separator:
                    if not found_first_separator:
                        # First separator - header is done
                        header_lines = current_section
                        found_first_separator = True
                    else:
                        # Save previous effect section
                        if current_section:
                            effect_sections.append(current_section)
                    current_section = []
                else:
                    current_section.append(line)

            # Don't forget the last section
            if current_section:
                effect_sections.append(current_section)

            # Parse header
            for line in header_lines:
                if line.startswith('Pipeline:'):
                    pipeline_name = line.split(':', 1)[1].strip()
                elif line.startswith('Description:'):
                    pipeline_description = line.split(':', 1)[1].strip()

            # Clear existing effects
            for frame in self.effect_frames[:]:
                frame.destroy()
            self.effect_frames = []

            for effect in self.effects:
                if hasattr(effect, 'cleanup'):
                    effect.cleanup()
            self.effects = []

            # Parse each effect section
            for section in effect_sections:
                if not section:
                    continue

                # First line is effect name
                effect_name = section[0].strip()

                # Find the effect by name
                effect_info = None
                for info in self.available_effects:
                    if info['name'] == effect_name:
                        effect_info = info
                        break

                if not effect_info:
                    print(f"Warning: Effect '{effect_name}' not found")
                    continue

                # Add the effect
                self._add_effect(effect_info, len(self.effects))
                effect = self.effects[-1]

                # Parse parameters from remaining lines
                # Skip description and method signature lines (they don't have ':' with values)
                for line in section[1:]:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        # Skip lines that look like method signatures
                        if '(' in key or ')' in key:
                            continue

                        # Generic parameter matching
                        self._set_effect_param_from_text(effect, key, value)

                # Re-render the effect's control panel to show updated values
                if hasattr(effect, '_toggle_mode'):
                    # Toggle twice to stay in same mode but refresh display
                    effect._toggle_mode()
                    effect._toggle_mode()

            # Set pipeline name and description
            self.pipeline_name.set(pipeline_name)
            self.pipeline_description.set(pipeline_description)

            # Update visibility of "Add First Effect" button
            if self.effects and self.add_first_frame:
                self.add_first_frame.pack_forget()

        except Exception as e:
            print(f"Error pasting text: {e}")

    def _set_effect_param_from_text(self, effect, key, value):
        """Set an effect parameter from a text key-value pair"""
        # Normalize key for matching
        key_lower = key.lower().replace(' ', '_')

        # First, check form schema for dropdown parameters
        # This handles cases where an IntVar is an index into a list of options
        if hasattr(effect, 'get_form_schema'):
            schema = effect.get_form_schema()
            for field in schema:
                field_label = field.get('label', '').lower().replace(' ', '_')
                field_key = field.get('key', '').lower()

                # Check if this schema field matches the key
                if key_lower in field_label or field_label in key_lower or key_lower == field_key:
                    if field.get('type') == 'dropdown' and 'options' in field:
                        options = field['options']
                        # Find the index of the value in options
                        for i, opt in enumerate(options):
                            opt_str = str(opt)
                            if opt_str == value:
                                # Found it! Now set the corresponding index variable
                                # The variable name is typically field_key + '_index' or just field_key
                                index_var_name = field_key + '_index'
                                if hasattr(effect, index_var_name):
                                    attr = getattr(effect, index_var_name)
                                    if isinstance(attr, tk.IntVar):
                                        attr.set(i)
                                        # Also update subform if exists
                                        if hasattr(effect, '_subform') and effect._subform and field_key in effect._subform._vars:
                                            effect._subform._vars[field_key].set(value)
                                        return
                                # Also try just the key (some effects use the key directly)
                                if hasattr(effect, field_key):
                                    attr = getattr(effect, field_key)
                                    if isinstance(attr, tk.StringVar):
                                        attr.set(value)
                                        return
                        # If dropdown options are numeric, try direct match
                        break

        # Fall back to direct attribute matching
        for attr_name in dir(effect):
            if attr_name.startswith('_'):
                continue

            # Check if this attribute name matches
            attr_lower = attr_name.lower()
            if key_lower in attr_lower or attr_lower in key_lower:
                try:
                    attr = getattr(effect, attr_name)
                    if isinstance(attr, tk.BooleanVar):
                        attr.set(value.lower() in ('yes', 'true', '1', 'on'))
                    elif isinstance(attr, tk.IntVar):
                        # Extract number from value
                        num = ''.join(c for c in value if c.isdigit() or c == '-')
                        if num:
                            attr.set(int(num))
                    elif isinstance(attr, tk.DoubleVar):
                        num = ''.join(c for c in value if c.isdigit() or c in '.-')
                        if num:
                            attr.set(float(num))
                    elif isinstance(attr, tk.StringVar):
                        attr.set(value)
                    return  # Found a match, stop looking
                except Exception as e:
                    pass  # Continue trying other attributes

    def _paste_json(self):
        """Paste pipeline from JSON on clipboard"""
        if not self.root:
            return

        try:
            text = self.root.clipboard_get()
            config = json.loads(text)

            # Store current mode to restore after re-render
            was_edit_mode = self._current_mode == 'edit'

            # Clear existing effects and effect frames
            for frame in self.effect_frames[:]:
                frame.destroy()
            self.effect_frames = []

            # Clear existing effects list (but don't call _remove_effect since frames are already destroyed)
            for effect in self.effects:
                if hasattr(effect, 'cleanup'):
                    effect.cleanup()
            self.effects = []

            # Load the pipeline configuration
            for effect_config in config.get('effects', []):
                # Find effect info
                for info in self.available_effects:
                    if info['module'] == effect_config['module'] and \
                       info['class_name'] == effect_config['class_name']:
                        # Add the effect
                        self._add_effect(info, len(self.effects))

                        # Restore parameters
                        effect = self.effects[-1]
                        for param_name, value in effect_config.get('params', {}).items():
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

            # Set name and description
            self.pipeline_name.set(config.get('name', ''))
            self.pipeline_description.set(config.get('description', ''))

            # Update visibility of "Add First Effect" button
            if self.effects and self.add_first_frame:
                self.add_first_frame.pack_forget()
            elif not self.effects and self.add_first_frame and was_edit_mode:
                self.add_first_frame.pack(fill='x', padx=10, pady=5)

        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def cleanup(self):
        """Cleanup all effects in the pipeline"""
        for effect in self.effects:
            if hasattr(effect, 'cleanup'):
                effect.cleanup()
