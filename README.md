# Webcam Filters - Unified Plugin System

A flexible, plugin-based webcam filter system featuring real-time video effects using Python, OpenCV, MediaPipe, and signal processing techniques.

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy mediapipe pillow matplotlib

# Run with default effect (passthrough)
python main.py

# Run with specific effect
python main.py seasonal/christmas
python main.py signals/fft_ringing
python main.py matrix/color

# List all available effects
python main.py --list

# List available cameras
python main.py --list-cameras

# Specify camera and resolution
python main.py seasonal/winter --camera 0 --width 1280 --height 720
```

## Table of Contents

- [Architecture](#architecture)
- [User Interface](#user-interface)
- [Global Controls](#global-controls)
- [Available Effects](#available-effects)
- [Creating New Effects](#creating-new-effects)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Technical Details](#technical-details)
- [Repository History](#repository-history)

---

## Architecture

### Unified Plugin System

The project uses a **plugin-based architecture** where all effects are organized into categories:

```
webcam-filters/
├── main.py                    # Main launcher
├── core/                      # Core framework
│   ├── base_effect.py        # Base classes for effects
│   ├── camera.py             # Camera utilities
│   └── video_window.py       # Tkinter video display
└── effects/                   # All effects organized by category
    ├── seasonal/             # Seasonal themes
    ├── matrix/               # Matrix-style effects
    ├── lines/                # Edge detection effects
    ├── refraction/           # Optical distortion
    ├── signals/              # Signal processing (FFT)
    ├── opencv/               # OpenCV operations
    └── misc/                 # Other effects
```

### Effect Discovery

Effects are **automatically discovered** at runtime. Each effect:
- Extends `BaseEffect` or `BaseUIEffect`
- Defines metadata (name, description, category)
- Implements `update()` and `draw()` methods
- Can optionally provide a control panel UI

---

## User Interface

### Global Controls Window

Available for all effects:
- **Camera Selection**: Switch between available cameras
- **Resolution**: 640×480, 1024×768, 1280×720, 1920×1080
- **Mirror Flip**: Toggle horizontal flip
- **Effect Selection**: Choose from all available effects (restarts program)
- **Output Controls**:
  - **Gain**: 0.1× to 10× (logarithmic slider)
  - **Invert**: Invert colors
  - **Show Original**: Bypass all effects

### Effect-Specific Controls

Some effects (like FFT Ringing) provide additional control panels with:
- Mode selection
- Parameter adjustment
- Real-time visualization
- Collapsible sections

### Window Layout

Windows are automatically positioned:
- **Global Controls**: Top left
- **Effect Controls**: Below global controls (if applicable)
- **Video Output**: Top right
- **Visualization**: Below video (if applicable)

---

## Available Effects

### Seasonal Effects (`seasonal/`)

| Effect | Description |
|--------|-------------|
| **christmas** | Pine garland border with reflective ornaments and falling snow |
| **winter** | Arctic blue edges with heavy snowfall |
| **fall** | Autumn color palette with realistic falling leaves |
| **summer** | Golden sunset with thermal heat wave distortion |
| **spring** | Spring-themed effects |

### Matrix Effects (`matrix/`)

| Effect | Description |
|--------|-------------|
| **color** | Colorful Matrix rain with saturated colors |
| **green** | Classic green-on-black Matrix aesthetic |
| **old_moving** | Physics-based Matrix with per-character flow simulation |

### Line/Edge Effects (`lines/`)

| Effect | Description |
|--------|-------------|
| **canny** | Basic Canny edge detection |
| **color_dense** | 24-layer RGB bit-plane edge decomposition |
| **mono_24_channels** | Pen-and-ink style from grayscale bit planes |
| **mono_traditional** | Stable sketch with temporal smoothing |
| **sketch** | Sketch-style line rendering |

### Refraction/Optical Effects (`refraction/`)

| Effect | Description |
|--------|-------------|
| **cut_glass** | Prismatic cut glass refraction |
| **rain_drops** | Realistic water droplet refraction (1000 drops) |
| **square_lenses** | Compound fisheye lens grid |

### Signal Processing (`signals/`)

| Effect | Description |
|--------|-------------|
| **fft_ringing** | Advanced FFT frequency filtering with 4 output modes:<br>• Grayscale composite<br>• Individual RGB channels<br>• Grayscale bit planes (8 layers)<br>• Color bit planes (24 layers)<br>Includes real-time filter visualization |
| **frequency_filter** | High-pass frequency filter |

### OpenCV Operations (`opencv/`)

A comprehensive collection of OpenCV image processing operations with interactive UI controls. These can be chained together using the **[Custom Pipelines](#custom-pipelines)** feature described below.

#### Edge Detection
| Effect | Description |
|--------|-------------|
| **edges_canny** | Classic Canny edge detection |
| **edges_scharr** | Scharr gradient (optimized 3x3, X/Y/Both directions) |
| **edges_laplacian** | Laplacian second derivative edge detection |

#### Morphological Operations
| Effect | Description |
|--------|-------------|
| **morph_erode** | Shrink bright regions |
| **morph_dilate** | Expand bright regions |
| **morph_open** | Erosion then dilation (remove small bright spots) |
| **morph_close** | Dilation then erosion (fill small dark holes) |

#### Feature Detection
| Effect | Description |
|--------|-------------|
| **detect_lines** | Hough line detection (standard and probabilistic) |
| **detect_circles** | Hough circle detection |
| **detect_corners_harris** | Harris corner detection |
| **detect_corners_shi_tomasi** | Shi-Tomasi corner detection |
| **contours** | Contour detection with 15 sorting options and 7 drawing modes |

#### Filters & Smoothing
| Effect | Description |
|--------|-------------|
| **bilateral_filter** | Edge-preserving smoothing |
| **median_blur** | Noise reduction with median filter |
| **mean_shift_filter** | Color segmentation (posterization-like effect) |

#### Color & Segmentation
| Effect | Description |
|--------|-------------|
| **color_inrange** | Filter colors by HSV/BGR range (mask, masked, inverse output) |
| **connected_components** | Blob labeling with psychedelic color visualization |

#### Transforms
| Effect | Description |
|--------|-------------|
| **scale_and_warp** | Translate, rotate, scale with border modes |

#### Other
| Effect | Description |
|--------|-------------|
| **blobs** | Blob detection with SimpleBlobDetector |

#### Custom Pipelines

The OpenCV category includes a powerful **Pipeline Builder** that lets you chain multiple effects together:

| Effect | Description |
|--------|-------------|
| **pipeline_builder** | Create custom effect chains with multiple OpenCV operations |

**Creating a Pipeline:**
- Select "opencv/(new user pipeline)" from the effect dropdown
- Use the "+" button to add effects to the chain
- Configure each effect's parameters
- Name your pipeline and click "Save"
- Click "Done Editing" to switch to view mode

**Pipeline Features:**
- Chain any OpenCV effects together (blur → edge detection → threshold, etc.)
- Reorder effects with drag handles
- Enable/disable individual effects in the chain
- Save multiple pipelines with custom names
- Edit saved pipelines anytime

**View Mode vs Edit Mode:**
- **Edit Mode**: Full control panels for each effect, add/remove/reorder effects
- **View Mode**: Read-only summary of effect settings with "Copy Settings" buttons

Saved pipelines appear in the effect dropdown as `opencv/user_<name>` and can be loaded directly.

**Note on Standalone Effects:** When running any OpenCV effect by itself (not in a pipeline), the control panel is always in "edit mode" with live, interactive controls. These settings are **not persisted** - the rationale is that most effects have few parameters and it's convenient to adjust them in real-time. Use pipelines when you want to save specific configurations.

### Miscellaneous (`misc/`)

| Effect | Description |
|--------|-------------|
| **passthrough** | The *default* effect - displays original camera feed |
| **stained_glass** | K-means color quantization |

---

## Creating New Effects

### Simple Effect Example

```python
# effects/my_category/my_effect.py
from core.base_effect import BaseEffect
import cv2

class MyEffect(BaseEffect):
    """My custom effect description"""

    @staticmethod
    def get_name():
        return "My Effect"

    @staticmethod
    def get_description():
        return "Does something cool"

    @staticmethod
    def get_category():
        return "my_category"

    def __init__(self, width, height):
        super().__init__(width, height)
        # Initialize effect parameters

    def update(self):
        """Update animation state"""
        pass

    def draw(self, frame, face_mask=None):
        """Apply effect to frame"""
        result = frame.copy()
        # Process frame here
        return result
```

The effect will be **automatically discovered** - no registration required!

---

## Global Controls

### Camera and Resolution

- **Camera**: Auto-detected, highest index selected by default
- **Resolution**: Changes take effect immediately (no restart)
- **Mirror Flip**: Horizontal flip for natural mirror view

### Output Processing

Applied to all effects:

**Gain** (0.1× to 10×)
- Logarithmic scale for fine control
- 1× = no change
- <1× = darker, >1× = brighter
- Applied after effect processing

**Invert**
- Inverts all color values (255 - value)
- Applied after gain

**Show Original**
- Bypasses all effects
- Useful for comparison

### Settings Persistence

All settings saved to `~/.webcam_filters_settings.json`:
- Last used effect
- Camera selection
- Flip preference
- Gain and invert state
- Effect-specific parameters

---

## Keyboard Shortcuts

- **SPACE**: Toggle effect on/off
- **Q** or **ESC**: Quit application
- **Ctrl+C**: Force exit (terminal)

---

## Technical Details

### Core Architecture

**BaseEffect**
- Simple effects without UI
- `update()`: Animation state
- `draw(frame)`: Process and return frame
- `cleanup()`: Resource cleanup

**BaseUIEffect**
- Effects with control panels
- Extends BaseEffect
- `create_control_panel(parent)`: Return Tkinter widget
- Access to root window for advanced UI

**UI Effect Example**

```python
from core.base_effect import BaseUIEffect
import tkinter as tk
from tkinter import ttk

class MyUIEffect(BaseUIEffect):
    """Effect with UI controls"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)
        self.param = tk.IntVar(value=50)

    def create_control_panel(self, parent):
        """Create Tkinter control panel"""
        panel = ttk.Frame(parent)

        ttk.Label(panel, text="Parameter:").pack()
        ttk.Scale(panel, from_=0, to=100,
                 variable=self.param).pack()

        return panel

    # ... implement draw(), etc.
```

**Effect Discovery**
- Automatic scanning of `effects/` directory
- No manual registration required
- Organizes by category
- Returns (key, name, description, category) tuples

### Video Pipeline

1. **Camera Capture**: OpenCV VideoCapture
2. **Mirror Flip**: Optional horizontal flip
3. **Effect Processing**:
   - `update()` - animate
   - `draw(frame)` - process
4. **Global Adjustments**: Gain → Invert
5. **Display**: Tkinter window with PIL/ImageTk

### Performance Optimizations

**Downsampling** (Fall, Stained Glass)
- Process at 33% resolution (1/9 pixels)
- 9× speed improvement
- Upsample result

**Pre-computation** (Rain Drops, Summer, Christmas)
- Generate displacement maps once
- Pre-calculate refraction patterns
- Store animated element properties

**Temporal Smoothing** (Mono Traditional)
- Frame blending (0.3 new, 0.7 old)
- Reduces flicker from auto-exposure

**Vectorization** (Square Lenses)
- NumPy array operations
- Advanced indexing for bulk pixel sampling

### Signal Processing (FFT Effects)

**Butterworth Filter**
```
H(u,v) = 1 / (1 + (D/D₀)^(2n))
```
- Smooth frequency cutoff
- Configurable radius and order
- Reduces ringing artifacts

**Bit-Plane Decomposition**
- Extract individual bit planes (MSB to LSB)
- Process each independently
- Recombine with weighting
- Reveals data structure

**Output Modes**
1. **Grayscale Composite**: Single channel filtering
2. **RGB Channels**: Per-channel filtering (creative color separation)
3. **Grayscale Bit Planes**: 8 independent filters
4. **Color Bit Planes**: 24 independent filters (8 per channel)

---

## Repository History

### Unified Architecture (Current)

**Main Branch**: Plugin-based system with all effects in organized directories

**Old Branches** (archived with `-old` suffix):
- Each effect was a separate branch
- Single `webcam_filter.py` per branch
- 20+ branches total
- Renamed for historical reference

### Migration Path

The repository evolved from:
1. **Single-branch approach**: All effects in separate branches
2. **Unified architecture**: All effects as plugins in main branch
3. **Branch cleanup**: Old branches renamed to `-old`

To view old implementations:
```bash
git checkout seasons-christmas-old  # View old single-file version
git checkout main                    # Return to unified system
```

---

## Dependencies

**Core Requirements**
- `opencv-python`: Computer vision and image processing
- `numpy`: Array operations and mathematical functions
- `mediapipe`: Face detection
- `pillow`: Image conversion for Tkinter display
- `matplotlib`: Real-time filter visualization (FFT effects)

**Python Version**: 3.7 or higher

**Operating Systems**: macOS, Linux, Windows

---

## Common Technical Patterns

### 1. Pre-generation for Consistency
Generate animated elements in `__init__()` for consistent appearance and better performance.

### 2. Performance via Downsampling
```python
small = cv2.resize(frame, (w // 3, h // 3))
processed = expensive_operation(small)
result = cv2.resize(processed, (w, h))
```

### 3. Edge Detection Pipeline
1. Convert to grayscale
2. Gaussian blur (5×5)
3. Canny edge detection
4. Optional dilation/blur
5. Alpha blend onto frame

### 4. Displacement Mapping
Pre-compute offset maps for realistic refraction effects.

### 5. Saturation Manipulation
```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=2.5)
result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

---

## Development

### Project Location
`~/Dropbox/dev/webcam-filters`

### Contributing

1. Create a new branch for your effect:
   ```bash
   git checkout -b my-new-effect
   ```

2. Add effect to appropriate category in `effects/`

3. Effect will be auto-discovered on next run

4. No need to modify `main.py` or any core files

---

## License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.

---

*Last updated: 2025-11-20*
*Architecture: Unified Plugin System*
*Total Effects: 40+*
