"""
Form renderer for consistent effect control panels.

Renders effect parameters as either interactive controls (edit mode),
read-only text (view mode), or empty fields (new mode).
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional


class Subform:
    """A subsection form for an effect's parameters.

    Each effect defines its schema and the Subform renders it
    in the appropriate mode with consistent table layout.

    Schema fields should include:
    - type: 'slider', 'dropdown', 'checkbox', 'radio', 'label', 'section'
    - label: Display label for the field
    - key: Key name for data access
    - default: Default value (optional)
    - For sliders: min, max
    - For dropdowns/radio: options
    """

    def __init__(self, schema: List[Dict[str, Any]]):
        """
        Args:
            schema: List of field definitions with keys, types, defaults, etc.
        """
        self.schema = schema
        self.frame = None
        self._widgets = {}  # Maps key -> widget
        self._vars = {}  # Maps key -> tk.Variable (for edit/new modes)

    def get_defaults(self) -> Dict[str, Any]:
        """Get default values from schema.

        Returns:
            Dictionary of key -> default value
        """
        defaults = {}
        for field in self.schema:
            key = field.get('key')
            if key:
                defaults[key] = field.get('default')
        return defaults

    def get_values(self) -> Dict[str, Any]:
        """Get current values from the form.

        Returns:
            Dictionary of key -> current value
        """
        values = {}
        for key, var in self._vars.items():
            try:
                values[key] = var.get()
            except:
                values[key] = None
        return values

    def render(self, parent, mode: str = 'edit', data: Dict[str, Any] = None) -> ttk.Frame:
        """Render the subform into the parent widget.

        Args:
            parent: Parent Tkinter widget
            mode: 'edit' (controls), 'view' (read-only), or 'new' (empty/default fields)
            data: Values to display/edit. If None, uses schema defaults.

        Returns:
            The frame containing the rendered form
        """
        self.frame = ttk.Frame(parent)
        self._widgets = {}
        self._vars = {}

        # Merge defaults with provided data
        values = self.get_defaults()
        if data:
            values.update(data)

        # Use grid layout for table alignment
        # Column 0: labels (right-justified)
        # Column 1: controls/values (left-justified)

        for row_idx, field in enumerate(self.schema):
            field_type = field.get('type', 'label')
            label_text = field.get('label', '')
            key = field.get('key', '')

            # Create label (right-justified)
            if label_text:
                label = ttk.Label(
                    self.frame,
                    text=f"{label_text}:",
                    anchor='e'
                )
                label.grid(row=row_idx, column=0, sticky='e', padx=(5, 10), pady=4)

            # Get value for this field
            value = values.get(key)

            # Create control/value based on type and mode
            if field_type == 'slider':
                self._render_slider(row_idx, field, mode, value)
            elif field_type == 'dropdown':
                self._render_dropdown(row_idx, field, mode, value)
            elif field_type == 'checkbox':
                self._render_checkbox(row_idx, field, mode, value)
            elif field_type == 'radio':
                self._render_radio(row_idx, field, mode, value)
            elif field_type == 'label':
                self._render_label_only(row_idx, field)
            elif field_type == 'section':
                self._render_section(row_idx, field, mode, values)

        # Configure column weights for proper sizing
        self.frame.columnconfigure(0, weight=0)  # Labels don't expand
        self.frame.columnconfigure(1, weight=1)  # Controls expand

        return self.frame

    def _render_slider(self, row: int, field: Dict[str, Any], mode: str, value):
        """Render a slider field"""
        key = field.get('key', '')
        min_val = field.get('min', 0)
        max_val = field.get('max', 100)

        if mode in ('edit', 'new'):
            # Create tk variable
            if isinstance(value, float):
                var = tk.DoubleVar(value=value if value is not None else 0)
            else:
                var = tk.IntVar(value=value if value is not None else 0)
            self._vars[key] = var

            # Container for slider and value label
            container = ttk.Frame(self.frame)
            container.grid(row=row, column=1, sticky='ew')

            # Slider
            slider = ttk.Scale(
                container,
                from_=min_val,
                to=max_val,
                orient='horizontal',
                variable=var
            )
            slider.pack(side='left', fill='x', expand=True)

            # Value label
            value_label = ttk.Label(container, text=str(int(var.get())), width=5)
            value_label.pack(side='left', padx=(5, 0))

            # Update value label when slider changes
            def update_label(*args, lbl=value_label, v=var):
                val = v.get()
                if isinstance(val, float):
                    lbl.config(text=f"{val:.1f}" if val != int(val) else str(int(val)))
                else:
                    lbl.config(text=str(val))
            var.trace_add('write', update_label)

            self._widgets[key] = slider

        else:  # view mode
            # Read-only text
            if value is not None:
                if isinstance(value, float):
                    text = f"{value:.1f}" if value != int(value) else str(int(value))
                else:
                    text = str(value)
            else:
                text = ''
            label = ttk.Label(self.frame, text=text)
            label.grid(row=row, column=1, sticky='w')

    def _render_dropdown(self, row: int, field: Dict[str, Any], mode: str, value):
        """Render a dropdown field"""
        key = field.get('key', '')
        options = field.get('options', [])

        if mode in ('edit', 'new'):
            # Create tk variable
            var = tk.StringVar(value=str(value) if value is not None else '')
            self._vars[key] = var

            combo = ttk.Combobox(
                self.frame,
                textvariable=var,
                values=[str(o) for o in options],
                state='readonly',
                width=20
            )
            combo.grid(row=row, column=1, sticky='w')

            # Set initial selection
            if value is not None and str(value) in [str(o) for o in options]:
                combo.set(str(value))

            self._widgets[key] = combo

        else:  # view mode
            text = str(value) if value is not None else ''
            label = ttk.Label(self.frame, text=text)
            label.grid(row=row, column=1, sticky='w')

    def _render_checkbox(self, row: int, field: Dict[str, Any], mode: str, value):
        """Render a checkbox field"""
        key = field.get('key', '')

        if mode in ('edit', 'new'):
            # Create tk variable
            var = tk.BooleanVar(value=bool(value) if value is not None else False)
            self._vars[key] = var

            cb = ttk.Checkbutton(self.frame, variable=var, text='')
            cb.grid(row=row, column=1, sticky='w')
            self._widgets[key] = cb

        else:  # view mode
            text = 'Yes' if value else 'No'
            label = ttk.Label(self.frame, text=text)
            label.grid(row=row, column=1, sticky='w')

    def _render_radio(self, row: int, field: Dict[str, Any], mode: str, value):
        """Render radio button options"""
        key = field.get('key', '')
        options = field.get('options', [])  # List of (value, text) tuples

        if mode in ('edit', 'new'):
            # Create tk variable
            var = tk.StringVar(value=str(value) if value is not None else '')
            self._vars[key] = var

            container = ttk.Frame(self.frame)
            container.grid(row=row, column=1, sticky='w')

            for opt_value, opt_text in options:
                rb = ttk.Radiobutton(
                    container,
                    text=opt_text,
                    variable=var,
                    value=str(opt_value)
                )
                rb.pack(side='left', padx=(0, 10))

            self._widgets[key] = container

        else:  # view mode
            # Find selected option text
            text = str(value) if value is not None else ''
            for opt_value, opt_text in options:
                if str(opt_value) == str(value):
                    text = opt_text
                    break
            label = ttk.Label(self.frame, text=text)
            label.grid(row=row, column=1, sticky='w')

    def _render_label_only(self, row: int, field: Dict[str, Any]):
        """Render a label-only field (no control)"""
        text = field.get('text', '')
        label = ttk.Label(self.frame, text=text)
        label.grid(row=row, column=1, sticky='w', pady=2)

    def _render_section(self, row: int, field: Dict[str, Any], mode: str, values: Dict[str, Any]):
        """Render a nested section (sub-subform)"""
        label_text = field.get('label', '')
        fields = field.get('fields', [])

        # Section label spans both columns
        if label_text:
            section_label = ttk.Label(
                self.frame,
                text=label_text,
                font=('TkDefaultFont', 10, 'bold')
            )
            section_label.grid(row=row, column=0, columnspan=2, sticky='w', pady=(10, 5))

        # Create nested subform
        if fields:
            nested = Subform(fields)
            nested_frame = nested.render(self.frame, mode, values)
            nested_frame.grid(row=row + 1, column=0, columnspan=2, sticky='ew', padx=(20, 0))


class EffectForm:
    """Container for an effect's form with enabled checkbox and buttons.

    This is the outer form that contains:
    - Effect name/title
    - Enabled checkbox
    - The effect's subform
    - Buttons column on the right (Edit/View, Copy Text, Copy JSON)
    """

    def __init__(self, effect_name: str, subform: Subform,
                 enabled_var: Optional[tk.BooleanVar] = None,
                 description: str = '', signature: str = '',
                 on_mode_toggle=None, on_copy_text=None, on_copy_json=None,
                 on_paste_text=None, on_paste_json=None,
                 on_add_below=None, on_remove=None):
        """
        Args:
            effect_name: Display name of the effect
            subform: The Subform containing the effect's parameters
            enabled_var: BooleanVar for the enabled checkbox
            description: Effect description text
            signature: Method signature text
            on_mode_toggle: Callback when Edit/View button is clicked
            on_copy_text: Callback when Copy Text button is clicked
            on_copy_json: Callback when Copy JSON button is clicked
            on_paste_text: Callback when Paste Text is needed
            on_paste_json: Callback when Paste JSON is needed
            on_add_below: Callback when + button is clicked (pipeline mode)
            on_remove: Callback when - button is clicked (pipeline mode)
        """
        self.effect_name = effect_name
        self.subform = subform
        self.enabled_var = enabled_var
        self.description = description
        self.signature = signature
        self.on_mode_toggle = on_mode_toggle
        self.on_copy_text = on_copy_text
        self.on_copy_json = on_copy_json
        self.on_paste_text = on_paste_text
        self.on_paste_json = on_paste_json
        self.on_add_below = on_add_below
        self.on_remove = on_remove
        self.frame = None
        self._mode_button = None
        self._current_mode = 'edit'

    def render(self, parent, mode: str = 'edit', data: Dict[str, Any] = None) -> ttk.Frame:
        """Render the complete effect form.

        Args:
            parent: Parent Tkinter widget
            mode: 'edit', 'view', or 'new'
            data: Values to display/edit

        Returns:
            The frame containing the complete form
        """
        self._current_mode = mode

        # Create style for LabelFrame with no left margin on label
        style = ttk.Style()
        style.layout('LeftAligned.TLabelframe', [
            ('Labelframe.border', {'sticky': 'nswe', 'border': 1, 'children': [
                ('Labelframe.padding', {'sticky': 'nswe'})
            ]})
        ])
        style.configure('LeftAligned.TLabelframe.Label', padding=(0, 0, 0, 0))
        style.configure('LeftAligned.TLabelframe', labelmargins=0, borderwidth=1)
        style.map('LeftAligned.TLabelframe', relief=[('', 'groove')])

        # Create label widget with spacer for proper vertical spacing
        label_container = tk.Frame(parent, bd=0, highlightthickness=0, padx=0, pady=0)
        tk.Label(label_container, text=self.effect_name, bd=0, highlightthickness=0, padx=0, pady=0).pack(anchor='w', padx=0, pady=0)
        tk.Label(label_container, text=" ", font=('TkDefaultFont', 3), bd=0, highlightthickness=0, padx=0, pady=0).pack(anchor='w', padx=0, pady=0)

        self.frame = ttk.LabelFrame(parent, labelwidget=label_container, labelanchor='nw', style='LeftAligned.TLabelframe')

        # Create disabled button style with gray text
        style.configure('Disabled.TButton', foreground='gray')

        # Main content area with 3 columns: enabled, subform, buttons
        content_frame = ttk.Frame(self.frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Left column: Enabled label + checkbox (vertically centered)
        if self.enabled_var is not None:
            enabled_frame = ttk.Frame(content_frame)
            enabled_frame.pack(side='left', fill='y', padx=(5, 10))

            # Spacer above to center vertically
            ttk.Frame(enabled_frame).pack(expand=True)

            ttk.Label(
                enabled_frame,
                text='Enabled',
                font=('TkDefaultFont', 9)
            ).pack()

            ttk.Checkbutton(
                enabled_frame,
                text='',
                variable=self.enabled_var
            ).pack()

            # Pipeline +/- buttons (if in pipeline mode)
            if self.on_add_below or self.on_remove:
                btn_frame = ttk.Frame(enabled_frame)
                btn_frame.pack(pady=(2, 0))

                if self.on_add_below:
                    plus_btn = tk.Label(
                        btn_frame,
                        text="  + ",
                        relief='raised',
                        borderwidth=1,
                        padx=0,
                        pady=0,
                        cursor='arrow'
                    )
                    plus_btn.pack(side='left', padx=(0, 1))
                    plus_btn.bind('<Button-1>', lambda e: self.on_add_below())
                    self._create_tooltip(plus_btn, "Add effect below")

                if self.on_remove:
                    minus_btn = tk.Label(
                        btn_frame,
                        text="  - ",
                        relief='raised',
                        borderwidth=1,
                        padx=0,
                        pady=0,
                        cursor='arrow'
                    )
                    minus_btn.pack(side='left')
                    minus_btn.bind('<Button-1>', lambda e: self.on_remove())
                    self._create_tooltip(minus_btn, "Remove this effect")

            # Spacer below to center vertically
            ttk.Frame(enabled_frame).pack(expand=True)

        # Center column: Description, signature, and subform
        center_frame = ttk.Frame(content_frame)
        center_frame.pack(side='left', fill='both', expand=True)
        self._center_frame = center_frame  # Expose for custom content

        # Description at top of center column
        if self.description:
            desc_label = ttk.Label(
                center_frame,
                text=self.description,
                font=('TkDefaultFont', 10)
            )
            desc_label.pack(anchor='w', pady=(0, 2))

        # Method signature below description
        if self.signature:
            sig_label = ttk.Label(
                center_frame,
                text=self.signature,
                font=('TkFixedFont', 10)
            )
            sig_label.pack(anchor='w', pady=(0, 5))

        # Subform below description/signature
        subform_frame = self.subform.render(center_frame, mode, data)
        subform_frame.pack(fill='x')

        # Right column: buttons (vertically centered)
        buttons_frame = ttk.Frame(content_frame)
        buttons_frame.pack(side='right', fill='y', padx=(10, 5))

        # Spacer above to center vertically
        ttk.Frame(buttons_frame).pack(expand=True)

        # Save/Edit toggle button
        button_text = "  Save  " if mode == 'edit' else "   Edit   "
        self._mode_button = tk.Label(
            buttons_frame,
            text=button_text,
            relief='raised',
            borderwidth=1,
            padx=0,
            pady=1,
            cursor='arrow'
        )
        self._mode_button.pack(pady=2)
        self._mode_button.bind('<Button-1>', lambda e: self._on_mode_button_click())

        # Copy buttons row (CT and CJ)
        copy_btn_frame = ttk.Frame(buttons_frame)
        copy_btn_frame.pack(pady=2)

        copy_text_btn = tk.Label(
            copy_btn_frame,
            text="CT",
            relief='raised',
            borderwidth=1,
            padx=2,
            pady=0,
            cursor='arrow'
        )
        copy_text_btn.pack(side='left', padx=(0, 1))
        copy_text_btn.bind('<Button-1>', lambda e: self._on_copy_text_click())
        self._create_tooltip(copy_text_btn, "Copy Text")

        copy_json_btn = tk.Label(
            copy_btn_frame,
            text="CJ",
            relief='raised',
            borderwidth=1,
            padx=2,
            pady=0,
            cursor='arrow'
        )
        copy_json_btn.pack(side='left')
        copy_json_btn.bind('<Button-1>', lambda e: self._on_copy_json_click())
        self._create_tooltip(copy_json_btn, "Copy JSON")

        # Paste button
        if mode == 'edit':
            paste_btn = tk.Label(
                buttons_frame,
                text=" Paste ",
                relief='raised',
                borderwidth=1,
                padx=2,
                pady=0,
                cursor='arrow'
            )
            paste_btn.bind('<Button-1>', lambda e: self._on_paste_click())
        else:
            paste_btn = tk.Label(
                buttons_frame,
                text=" Paste ",
                relief='raised',
                borderwidth=1,
                padx=2,
                pady=0,
                cursor='arrow',
                fg='gray'
            )
        paste_btn.pack(pady=2)
        self._create_tooltip(paste_btn, "Paste (auto-detects format)")

        # Spacer below to center vertically
        ttk.Frame(buttons_frame).pack(expand=True)

        return self.frame

    def _on_mode_button_click(self):
        """Handle Edit/View toggle button click"""
        if self.on_mode_toggle:
            self.on_mode_toggle()

    def _on_copy_text_click(self):
        """Handle Copy Text button click"""
        if self.on_copy_text:
            self.on_copy_text()

    def _on_copy_json_click(self):
        """Handle Copy JSON button click"""
        if self.on_copy_json:
            self.on_copy_json()

    def _on_paste_click(self):
        """Handle Paste button click - auto-detects JSON or text format"""
        import json
        try:
            # Get clipboard content
            clipboard = self.frame.clipboard_get()

            # Try JSON first
            try:
                json.loads(clipboard)
                # It's valid JSON, use JSON paste
                if self.on_paste_json:
                    self.on_paste_json()
                    return
            except (json.JSONDecodeError, ValueError):
                pass

            # Fall back to text paste
            if self.on_paste_text:
                self.on_paste_text()
        except tk.TclError:
            # Clipboard empty or unavailable
            pass

    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief='solid', borderwidth=1)
            label.pack()
            widget._tooltip = tooltip

        def hide_tooltip(event):
            if hasattr(widget, '_tooltip') and widget._tooltip:
                widget._tooltip.destroy()
                widget._tooltip = None

        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)
