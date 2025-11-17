# Webcam Filter Effects - Development Synopsis

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Reference - Branch Switching](#quick-reference---branch-switching)
- [Installation & Setup](#installation--setup)
- [Repository Structure](#repository-structure)
- [Branch Descriptions](#branch-descriptions)
  - [Seasonal Effects](#seasonal-effects)
    - [Christmas](#branch-seasons-christmas)
    - [Winter](#branch-seasons-winter)
    - [Fall](#branch-seasons-fall)
    - [Summer](#branch-seasons-summer)
    - [Spring](#branch-seasons-spring)
  - [Matrix Effects](#matrix-effects)
    - [Color](#branch-matrix-color)
    - [Green](#branch-matrix-green)
    - [Physics Version](#branch-matrix-old-moving-char-physics-version)
  - [Line/Edge Effects](#lineedge-effects)
    - [Canny](#branch-lines-canny)
    - [Color Dense](#branch-lines-color-dense)
    - [Color Enhanced](#branch-lines-color-enhanced)
    - [Mono from 24 Channels](#branch-lines-mono-from-24-channels)
    - [Mono Traditional Canny](#branch-lines-mono-traditional-canny)
    - [Sketch](#branch-lines-sketch)
  - [Refraction/Optical Effects](#refractionoptical-effects)
    - [Cut Glass](#branch-refraction-cut-glass)
    - [Rain Drops](#branch-refraction-rain-drops)
    - [Square Lenses](#branch-refraction-square-lenses)
  - [Miscellaneous](#miscellaneous)
    - [Main (FFT Filtering)](#branch-main)
    - [Frequency Filter](#branch-frequency-filter)
    - [Signals Ringing](#branch-signals-ringing)
    - [Stained Glass](#branch-stained-glass)
- [Common Technical Patterns](#common-technical-patterns)
- [Key Learnings](#key-learnings)
- [Git Branch Summary](#git-branch-summary)
- [Performance Metrics](#performance-metrics)
- [Code Architecture](#code-architecture)
- [Dependencies](#dependencies)

---

## Project Overview
Development of seasonal-themed webcam filters using Python, OpenCV, MediaPipe, and NumPy. Multiple git branches created for different seasonal effects, all using a single `webcam_filter.py` file per branch.

---

## Quick Reference - Branch Switching

| Branch | Effect | Command |
|--------|--------|---------|
| **main** | FFT frequency filtering with multiple modes | `git checkout main` |
| **frequency-filter** | High-pass frequency filter | `git checkout frequency-filter` |
| **signals-ringing** | Signal processing with ringing effects | `git checkout signals-ringing` |
| **Seasonal Effects** |
| **seasons-christmas** | Pine garland with ornaments and snow | `git checkout seasons-christmas` |
| **seasons-winter** | Arctic blue edges with heavy snow | `git checkout seasons-winter` |
| **seasons-fall** | Autumn colors with falling leaves | `git checkout seasons-fall` |
| **seasons-summer** | Golden sunset with thermal heat waves | `git checkout seasons-summer` |
| **seasons-spring** | Spring-themed effects | `git checkout seasons-spring` |
| **stained-glass** | Color quantization effect | `git checkout stained-glass` |
| **Matrix Effects** |
| **matrix-color** | Colorful Matrix rain | `git checkout matrix-color` |
| **matrix-green** | Classic green Matrix rain | `git checkout matrix-green` |
| **matrix-old-moving-char-physics-version** | Physics-based Matrix flow | `git checkout matrix-old-moving-char-physics-version` |
| **Line/Edge Effects** |
| **lines-canny** | Basic Canny edge detection | `git checkout lines-canny` |
| **lines-color-dense** | 24-layer RGB bit-plane edges | `git checkout lines-color-dense` |
| **lines-color-enhanced** | Enhanced color line effects | `git checkout lines-color-enhanced` |
| **lines-mono-from-24-channels** | Pen-and-ink on white | `git checkout lines-mono-from-24-channels` |
| **lines-mono-traditional-canny** | Stable textured sketch | `git checkout lines-mono-traditional-canny` |
| **lines-sketch** | Sketch-style line rendering | `git checkout lines-sketch` |
| **Refraction/Optical Effects** |
| **refraction-cut-glass** | Cut glass refraction effect | `git checkout refraction-cut-glass` |
| **refraction-rain-drops** | Water droplet refraction | `git checkout refraction-rain-drops` |
| **refraction-square-lenses** | Compound fisheye grid | `git checkout refraction-square-lenses` |

**After switching branches, run:**
```bash
python webcam_filter.py
# Or for Python 3 specifically:
python3 webcam_filter.py
```

---

## Installation & Setup

### Prerequisites
- **Python Version**: Python 3.7 or higher recommended
- **Operating System**: macOS, Linux, or Windows
- **Webcam**: Built-in or USB webcam

### Installing Dependencies

```bash
# Install required packages
pip install opencv-python numpy mediapipe pillow matplotlib

# Or use pip3 if you have multiple Python versions
pip3 install opencv-python numpy mediapipe pillow matplotlib
```

### Dependency Details
- **opencv-python** (cv2): Image processing and computer vision
- **numpy**: Array operations and mathematical functions
- **mediapipe**: Face detection (used in some effects)
- **pillow** (PIL): Image format conversion for Tkinter display
- **matplotlib**: Real-time filter visualization (main branch)

### Camera Setup

#### Finding Available Cameras
The application automatically detects available cameras on startup. You'll see a list like:
```
Available cameras:
0: Built-in Camera
1: USB Webcam
```

#### Troubleshooting Camera Issues

**Camera not detected:**
- Check camera permissions in System Preferences (macOS) or Settings (Windows)
- Ensure no other application is using the camera
- Try unplugging and replugging USB cameras

**Poor performance:**
- Close other applications using the camera
- Try a lower-resolution camera if available
- Some effects (rain-drops, matrix-physics) are more demanding

**Camera shows black screen:**
- Check if camera lens is covered
- Verify camera works in other applications
- Try a different camera index (0, 1, 2, etc.)

### Running the Application

```bash
# Navigate to the repository
cd /path/to/webcam-filters

# Switch to desired branch
git checkout christmas

# Run the application
python webcam_filter.py
```

### Keyboard Controls
- **SPACEBAR**: Toggle effect on/off
- **Q or ESC**: Quit application
- **Ctrl+C**: Force exit (terminal)

---

## Repository Structure
- **Main repository**: `/Users/mbennett/Dropbox/dev/webcam-filters`
- **Single file approach**: `webcam_filter.py` - different versions on different branches
- **Git branches** (15 total):
  - **Seasonal**: christmas, winter, fall, summer
  - **Matrix variants**: matrix-color, matrix-green, matrix-old-moving-char-physics-version
  - **Edge/Line effects**: color-dense-lines, mono-lines-from-24-channels, mono-traditional-canny-lines
  - **Other effects**: stained-glass, rain-drops, square-lenses
  - **Base**: main

---

## Branch Descriptions

All webcam filter effects available in this repository, organized by category.

---

### Seasonal Effects

Seasonal-themed filters for different times of year.

---

## Branch Seasons-Christmas

```bash
git checkout seasons-christmas
```

### Effect Description
Pine garland border with reflective ornamental balls and falling snow.

### Key Features
1. **Pine Garland Border**
   - Covers top, bottom, left, and right edges
   - Wavy pattern using sine waves for natural appearance
   - Depth of 50 pixels at edges

2. **Ornamental Balls** (lines 31-84)
   - Pre-generated in `__init__()` for consistent appearance
   - Three colors: deep red (0, 0, 200), deep green (0, 180, 0), deep gold (0, 180, 220)
   - Sizes: 12-18 pixels
   - Reflective appearance with:
     - Gradient shading (darker edges to brighter center)
     - Colored highlight spot
     - White specular highlight
     - Darker outline for depth
   - Fixed positions along garland borders

3. **Snow Effect**
   - Irregular polygon shapes (not circles)
   - Falling animation with drift

### Technical Implementation
```python
# Ornament ball rendering with reflective appearance
for ball in self.ornament_balls:
    # Gradient shading
    # Colored highlight
    # White specular highlight
    # Dark outline
```

### Critical Fix
**Problem**: Ornament balls were changing color every frame
**Cause**: Random generation in `draw()` method
**Solution**: Moved generation to `__init__()` with fixed positions, sizes, and colors

---

## Branch Seasons-Winter

```bash
git checkout seasons-winter
```

### Effect Description
Arctic blue near edges transitioning to white in center, with falling snow.

### Key Features
1. **Edge Detection and Processing** (lines 76-96)
   - Canny edge detection
   - Progressive Gaussian blur cycles (no dilation to avoid blockiness)
   - 12 blur cycles with increasing kernel sizes: 15×15, 21×21, 27×27, 33×33, 39×39, 45×45, 51×51, 57×57, 63×63, 71×71, 81×81, 91×91
   - Additional 151×151 blur with 1.3× boost for final feathering
   - Each blur cycle includes 1.15× opacity boost to maintain strength

2. **Color Application**
   - Inverted edge mask (255 - edges)
   - Arctic blue (BGR: 255, 255, 180) near edges
   - White away from edges
   - Smooth blending using alpha compositing

3. **White Edge Highlights**
   - Additional white edges drawn on top
   - Slightly blurred for soft appearance

4. **Snow**
   - Irregular white polygon shapes (3-5 points)
   - Sizes: 1.5-4.0 pixels
   - Falling speed: 2.0-6.0 pixels/frame
   - Horizontal drift: -0.5 to 0.5
   - 300 snowflakes for heavy snow effect

### Technical Implementation
```python
# Progressive blur cycles - smooth, non-blocky edges
edges_processed = edges.copy().astype(np.float32)
blur_sizes = [(15, 15), (21, 21), ..., (91, 91)]
for blur_size in blur_sizes:
    edges_processed = cv2.GaussianBlur(edges_processed, blur_size, 0)
    edges_processed = np.clip(edges_processed * 1.15, 0, 255)

# Additional blur and boost
edges_blurred = cv2.GaussianBlur(edges_processed, (151, 151), 0)
edges_blurred = np.clip(edges_blurred * 1.3, 0, 255).astype(np.uint8)

# Invert for arctic blue near edges, white away
inverted_mask = 255 - edges_blurred
```

### Development Iterations
1. Initial approach: Dilation + blur → **too blocky**
2. Only Gaussian blurs (12 cycles) → **better but needed more coverage**
3. Tried 18 blur cycles → **too much coverage**
4. Settled on 12 cycles + additional 151×151 blur + 1.3× boost → **optimal**
5. Multiple power curve adjustments (3.0, 5.0, 2.0, back to 5.0)
6. Final approach: Remove power curve, use inverted mask with additional blur

### Debugging Approach
Created 2×2 grid visualization showing:
- Upper left: Canny edges
- Upper right: Dilated edges
- Lower left: Blurred edges (alpha mask)
- Lower right: Original frame

User feedback guided iterative refinement.

---

## Branch Seasons-Fall

```bash
git checkout seasons-fall
```

### Effect Description
Autumn color palette with realistic falling leaves.

### Key Features
1. **Performance Optimization** (lines 86-88)
   - Downsample to 33% (1/9 pixels) for 9× speed improvement
   - Process at lower resolution
   - Upsample back to original size
   ```python
   small_frame = cv2.resize(frame, (original_width // 3, original_height // 3))
   # Process...
   result = cv2.resize(result, (original_width, original_height))
   ```

2. **Fall Color Palette** (lines 90-107)
   - 5 shades of green: dark, forest, lime, olive, dark olive
   - Multiple oranges: dark orange, orange red, orange
   - Reds: dark red, maroon
   - Golds and yellows: gold, yellow, golden rod
   - Purples: dark magenta, purple
   - Blue pixels preserved (sky/water)

3. **Color Mapping Algorithm**
   - Calculate Euclidean distance in RGB space
   - Map each pixel to nearest fall color
   - Preserve blue pixels (B > G and B > R)

4. **Falling Leaves** (lines 19-40)
   - 15 leaves at various positions
   - Pointed teardrop shape with curved sides
   - Oscillating horizontal motion (sine wave)
   - Speed increased 3× from original: 4.5-10.5 pixels/frame
   - Rotation with variable speed: -3 to 3 degrees/frame
   - Sizes: 15-30 pixels
   - Small stem at top

### Technical Implementation
```python
# Realistic leaf shape (pointed teardrop)
vertices = np.array([
    [cx, cy - size/2],           # Top point
    [cx + size/3, cy],           # Right middle (curved)
    [cx + size/4, cy + size/2],  # Right bottom
    [cx, cy + size/2.2],         # Bottom point
    [cx - size/4, cy + size/2],  # Left bottom
    [cx - size/3, cy],           # Left middle (curved)
])

# Oscillating motion
leaf['x'] = base_x + oscillation_amplitude * np.sin(time * oscillation_speed + phase)
```

---

## Branch Seasons-Spring

```bash
git checkout seasons-spring
```

### Effect Description
Spring-themed webcam filter effects.

### Key Features
1. **Spring Elements**
   - Cherry blossoms, rain showers, or green tints
   - Seasonal color palette

---

## Branch Stained Glass

```bash
git checkout stained-glass
```

### Effect Description
Stained glass window effect using color quantization.

### Key Features
1. **Performance Optimization**
   - Same 33% downsampling technique as fall branch
   - 9× speed improvement
   - Critical for k-means clustering performance

2. **K-means Color Quantization**
   - Reduces colors to create stained glass segments
   - Bilateral filtering for smoothing while preserving edges

---

## Branch Frequency-Filter

```bash
git checkout frequency-filter
```

### Effect Description
High-pass frequency filtering for edge enhancement and detail extraction.

### Key Features
1. **Adjustable Blur Kernel**
   - Configurable blur kernel size
   - High-pass filter implementation

---

## Branch Signals-Ringing

```bash
git checkout signals-ringing
```

### Effect Description
Signal processing effects with ringing artifacts.

### Key Features
1. **Signal Processing**
   - Ringing effects from frequency filtering
   - Advanced signal analysis

---

### Miscellaneous

Other effects that don't fit into specific categories.

---

## Branch Main

```bash
git checkout main
```

### Effect Description
Simple white edges on original color background.

### Key Features
1. **Simplified from Matrix**
   - Removed all Matrix characters, streamers, and grid code
   - Kept only edge detection
   - White edges (255, 255, 255)
   - Original frame color preserved

2. **Edge Detection**
   - Canny algorithm (thresholds: 50, 150)
   - Gaussian blur (5×5)
   - Dilation (2×2 kernel, 1 iteration)
   - Soft edge effect

---

## Branch Seasons-Summer

```bash
git checkout seasons-summer
```

### Effect Description
Golden sunset with thermal heat waves rising from bottom.

### Key Features

#### 1. Golden Sunset Effect (lines 125-172)
**HSV Hue Replacement**
- Convert frame to HSV color space
- Replace all hues with golden sunset hue = 20 (0-179 scale)
- Preserve original saturation (S) and value (V)
```python
hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
golden_hue = 20
hsv[:, :, 0] = golden_hue
result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

**Gradient Overlay** (lines 136-156)
- Top half has gentle golden sunlight gradient
- Max strength: 0.5 at very top
- Linear fade to 0.0 at halfway down
- Bright golden color: BGR (100, 200, 255)
- Smoother transition than initial attempt
```python
if y < height // 2:
    gradient[y, :] = 0.5 * (1.0 - (y / (height // 2)))
else:
    gradient[y, :] = 0.0
```

**Golden Edges**
- Edge color: BGR (0, 165, 255) - bright golden orange
- Canny edge detection
- Gaussian blur for soft effect
- Alpha blending

#### 2. Thermal Heat Waves (NEW - lines 19-123)
**Initialization**
- 8 heat waves with precomputed refraction
- Wave dimensions:
  - Width: 1/3 to 1/2 of screen width
  - Height: 40-80 pixels
- Start position: Below screen, random x position

**Precomputed Refraction Displacement Maps** (lines 26-50)
- Lens-like distortion pattern
- Horizontal displacement: wave pattern using `sin(x * 0.1 + y * 0.05) * 8.0`
- Vertical displacement: slight upward distortion (3.0 pixels)
- Strength varies by distance from center (strongest at center)
- Gaussian blur (15×15) applied for smooth refraction

```python
displacement_map = np.zeros((wave_height, wave_width, 2), dtype=np.float32)
for y in range(wave_height):
    for x in range(wave_width):
        center_y = wave_height / 2.0
        dist_from_center = abs(y - center_y) / center_y
        strength = (1.0 - dist_from_center) * np.sin(dist_from_center * np.pi)
        x_offset = strength * np.sin(x * 0.1 + y * 0.05) * 8.0
        y_offset = strength * 3.0
        displacement_map[y, x] = [x_offset, y_offset]
```

**Animation** (lines 62-73)
- Move upward at 0.8-2.0 pixels/frame
- Reset below screen when disappearing near middle
- Continuous loop creating rising heat shimmer

**Rendering** (lines 75-123)
- Applied before golden sunset processing
- Only process visible portions for efficiency
- Per-pixel refraction using displacement maps
- Creates thermal distortion effect

```python
# Apply refraction
for y in range(region_h):
    for x in range(region_w):
        dx, dy = disp_map[y, x]
        src_x = int(x + dx)
        src_y = int(y + dy)
        # Clamp and copy refracted pixel
        result[dst_y_start + y, dst_x_start + x] = region[src_y, src_x]
```

### Development History
1. Created from main branch (white edges on color)
2. Added golden hue replacement (HSV = 20)
3. Changed edges to golden color
4. Added gradient overlay (initially too strong: 1.0 → 1/3 down)
5. Refined gradient (0.5 max strength → halfway down, smoother transition)
6. **LATEST**: Added thermal heat waves with precomputed refraction

---

### Matrix Effects

Classic Matrix-style rain effects with various color schemes and physics.

---

## Branch Matrix-Color

```bash
git checkout matrix-color
```

### Effect Description
Matrix-style character grid with full-color saturation, creating a vibrant, colorful Matrix rain effect with colored edges and character streamers.

### Key Features

1. **Saturation Boost** (lines 86-93)
   - Converts frame to HSV color space
   - Boosts saturation by 2.5× for vivid, vibrant colors
   - Preserves hue and value while amplifying color intensity

2. **Colored Edge Detection** (lines 98-112)
   - Canny edge detection with Gaussian blur softening
   - Edges masked with saturated frame colors
   - Edge overlay at 45% intensity creates colored outlines
   - Background dimmed to 30% to make edges and characters stand out

3. **Fixed Character Grid** (lines 28-37)
   - Grid spacing: 10px horizontal, 18px vertical
   - Characters change randomly every 20-40 frames
   - ASCII character set for maximum rendering speed

4. **Color-Aware Streamers** (lines 39-82, 136-178)
   - 30% of columns have active streamers
   - White heads (255, 255, 255) with thickness=2
   - Colored tails sample background color at each position
   - Tail intensity scales with background color and distance from head

### Technical Implementation
```python
# Saturation boost for vibrant colors
hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=2.5, beta=0)

# Characters sample color from saturated background
bg_color = saturated_frame[y_pos, x_pos]
intensity_scale = streamer_intensity / 200.0
color = tuple(int(c * intensity_scale * 1.5) for c in bg_color)
```

### Unique Features
- **Brightness-based character rendering**: Characters inherit color from the saturated background at their position
- **Dual-layer compositing**: Dimmed saturated background + colored edges + colored characters
- **Adaptive character coloring**: Streamer tails use boosted background colors, static grid uses dimmer colors

---

## Branch Matrix-Green

```bash
git checkout matrix-green
```

### Effect Description
Classic Matrix rain effect with traditional green-on-black aesthetic, featuring green characters and edges on a dark background.

### Key Features

1. **Pure Green Color Palette** (lines 86-108)
   - All color removed from original frame
   - Converts to grayscale, then maps only to green channel
   - Black background (BGR: 0, 0, 0)
   - Green edges and characters only

2. **Green Edge Background** (lines 87-108)
   - Canny edges at 45% intensity
   - Edges placed in green channel only
   - Very dim grayscale background (15% intensity) in green channel
   - Classic Matrix aesthetic

3. **White Streamer Heads** (lines 138-170)
   - Head of each streamer renders as white (final_intensity for all channels)
   - Creates bright highlight effect
   - Tail fades through green intensities

4. **Brightness Mapping** (lines 128-153)
   - Characters reflect background brightness at 70% intensity
   - Darker areas have no characters (threshold: 20)
   - Maintains contrast from original scene

### Technical Implementation
```python
# Convert to pure green on black
edge_background = np.zeros_like(frame)
edge_background[:, :, 1] = edges  # Green channel only

dimmed_frame = np.zeros_like(frame)
dimmed_frame[:, :, 1] = dimmed_gray  # Only green channel
```

### Unique Features
- **Monochromatic channel isolation**: All visual information mapped to single color channel
- **Brightness preservation**: Maintains scene brightness while removing all hue/saturation

### Comparison to matrix-color
- matrix-color: Vivid, colorful, modern interpretation
- matrix-green: Classic, traditional Matrix aesthetic

---

## Branch Matrix-Old-Moving-Char-Physics-Version

```bash
git checkout matrix-old-moving-char-physics-version
```

### Effect Description
Advanced Matrix rain with realistic physics-based character flow around obstacles (edges and faces). Characters flow like water particles, each with independent lateral movement.

### Key Features

1. **Individual Character Physics** (lines 37-38, 60-128)
   - Each character in a streamer has its own:
     - `char_x_offsets`: Independent lateral position offset
     - `char_x_velocities`: Independent lateral velocity
   - Characters behave like individual water droplets

2. **Collision Detection and Flow Physics** (lines 71-122)
   - Looks ahead 25 pixels for collision detection
   - Detects horizontal vs vertical edges
   - Searches up to 90 pixels left/right for flow direction
   - Applies strong forces: 20.0 for horizontal edges, 15.0 for vertical

3. **Fluid Dynamics Simulation** (lines 113-128)
   - Force application based on available space
   - Damping coefficient: 0.92 (allows sustained flow)
   - Velocity clamp: ±250 pixels for dramatic movement
   - Natural-looking flow around obstacles

4. **Ultra-Dense Streamer Grid** (lines 22-24)
   - Column width: 4 pixels (ultra-narrow)
   - Maximum number of streamers for dense rain
   - 30% faster speeds: 3.29-13.16 pixels/frame

5. **Green Gradient Fade** (lines 166-184)
   - White head (220, 255, 255)
   - Bright green (100, 255, 200) at position 1
   - Progressive fade through 10+ intensity levels
   - Tail fades to very dark green (0, 5, 0)

6. **Enhanced Background** (lines 188-197)
   - 50% brightness original frame (not heavily dimmed)
   - Green tint overlay at 25% strength
   - Better visibility of underlying scene

### Technical Implementation
```python
# Independent character physics
for i in range(len(drop['chars'])):
    char_y = int(drop['y'] - (i * self.char_spacing))
    char_x = int(drop['x'] + drop['char_x_offsets'][i])

    # Collision detection and force calculation
    if collision:
        force = 20.0 if is_horizontal_edge else 15.0
        if left_space > right_space:
            drop['char_x_velocities'][i] -= force
        else:
            drop['char_x_velocities'][i] += force

    # Apply velocity with damping
    drop['char_x_offsets'][i] += drop['char_x_velocities'][i]
    drop['char_x_velocities'][i] *= 0.92
```

### Unique Features
- **Per-character physics**: Each character is an independent particle with position and velocity
- **Predictive collision detection**: Looks ahead to anticipate obstacles
- **Asymmetric flow detection**: Searches for open space and applies directional forces
- **Water-like behavior**: Characters flow naturally around obstacles like liquid

### Performance Optimizations
- Early exit for offscreen drops
- Only process visible characters
- Pre-allocated character arrays

### Complexity Note
This is the most sophisticated implementation with realistic physics simulation.

---

### Line/Edge Effects

Edge detection and line-based artistic rendering effects.

---

## Branch Lines-Canny

```bash
git checkout lines-canny
```

### Effect Description
Basic Canny edge detection for clean line art effects.

### Key Features
1. **Canny Edge Detection**
   - Standard Canny algorithm
   - Adjustable thresholds

---

## Branch Lines-Color-Dense

```bash
git checkout lines-color-dense
```

### Effect Description
Artistic edge visualization using bit-plane decomposition across RGB channels, creating a dense, colorful layered line effect.

### Key Features

1. **Multi-Channel Bit-Plane Analysis** (lines 117-172)
   - Processes each RGB channel separately
   - Extracts all 8 bit planes (MSB to LSB) per channel
   - Total: 24 edge layers (3 channels × 8 bits)

2. **Bit Plane Extraction** (lines 133-136)
   - Uses bit shifting to isolate each bit plane
   - `(channel_data >> bit) & 1` extracts single bit
   - Converts binary plane to 255-value image for edge detection

3. **Hierarchical Edge Thickness** (lines 142-146)
   - MSB planes (bits 6-7): 3×3 dilation, 1 iteration (thicker lines)
   - LSB planes (bits 0-5): No dilation (thinner lines)
   - Creates natural thickness variation

4. **Intensity Gradient** (lines 151-156)
   - Linear falloff: `intensity = (bit + 1) / 8.0`
   - Overall reduction: 50% intensity for all edges
   - Prevents oversaturation while maintaining detail

5. **Transparent Color Blending** (lines 158-167)
   - Floating-point accumulator for smooth blending
   - Each channel gets its own color: Blue (1,0,0), Green (0,1,0), Red (0,0,1)
   - Edges from different channels blend additively

6. **Saturated Background** (lines 86-112)
   - 2.5× saturation boost
   - 50% dimmed background (more visible than matrix versions)
   - Colored edge overlay at 45% intensity

### Technical Implementation
```python
# Bit plane extraction and edge detection
for channel_data, color_mask in channels:
    for bit in range(7, -1, -1):
        bit_plane = ((channel_data >> bit) & 1) * 255
        edges = cv2.Canny(bit_plane, 50, 150)

        # MSB gets thicker lines
        if bit >= 6:
            edges = cv2.dilate(edges, kernel, iterations=1)

        # Intensity based on bit significance
        intensity = (bit + 1) / 8.0 * 0.5

        # Add to color accumulator
        for c in range(3):
            edge_accumulator[:, :, c] += edges_float * color_mask[c]
```

### Unique Features
- **Bit-plane edge decomposition**: Novel approach to extracting edges at multiple intensity levels
- **Multi-channel additive blending**: RGB channels processed independently and combined
- **Significance-based weighting**: MSB planes contribute more (thicker, brighter) than LSB

### Use Case
Artistic visualization showing how color information is encoded in digital images. Reveals underlying data structure while creating aesthetically pleasing effect.

---

## Branch Lines-Color-Enhanced

```bash
git checkout lines-color-enhanced
```

### Effect Description
Enhanced color-based line detection with vibrant edge effects.

### Key Features
1. **Enhanced Color Processing**
   - Saturated color edges
   - Vibrant line rendering

---

## Branch Lines-Mono-From-24-Channels

```bash
git checkout lines-mono-from-24-channels
```

### Effect Description
Pen-and-ink style rendering using bit-plane decomposition from grayscale, creating dark lines on white background like a traditional sketch.

### Key Features

1. **Inverted Aesthetic** (lines 92-93)
   - White background (255, 255, 255)
   - Dark lines (approaching black)
   - Pen-and-ink or woodcut print style

2. **Grayscale Bit-Plane Processing** (lines 102-145)
   - Processes single grayscale channel
   - Extracts all 8 bit planes
   - Runs Canny on each bit plane independently
   - Total: 8 edge layers (despite "24-channels" in name)

3. **Thickness Hierarchy** (lines 111-114)
   - MSB planes (bits 6-7): 3×3 dilation
   - LSB planes (bits 0-5): No dilation
   - Creates varied line weights like hand-drawn sketch

4. **Darkness Scaling** (lines 117-124)
   - Linear falloff: `intensity = (bit + 1) / 8.0`
   - 80% overall darkness
   - Accumulates edge darkness in floating-point

5. **Inversion for White Background** (lines 139-146)
   - Edge accumulator represents darkness
   - Inverts: `edges_inverted = 255 - edges_grayscale`
   - Subtracts from white background

### Technical Implementation
```python
# Process grayscale bit planes
for bit in range(7, -1, -1):
    bit_plane = ((gray >> bit) & 1) * 255
    edges = cv2.Canny(bit_plane, 50, 150)

    # MSB gets thickness
    if bit >= 6:
        edges = cv2.dilate(edges, kernel, iterations=1)

    intensity = (bit + 1) / 8.0 * 0.8
    edge_accumulator += edges_float * intensity

# Invert for white background
edges_inverted = 255 - edges_grayscale
result = cv2.subtract(white_background, 255 - edges_3channel)
```

### Unique Features
- **Grayscale bit-plane decomposition**: Reveals tonal structure
- **Accumulated darkness**: Builds up line darkness from multiple planes
- **Inverted rendering**: Dark lines on white (opposite of typical effects)

### Artistic Style
Creates traditional illustration effect - resembles pen and ink drawings, etchings, or woodcut prints.

---

## Branch Lines-Mono-Traditional-Canny

```bash
git checkout lines-mono-traditional-canny
```

### Effect Description
Traditional Canny edge detection with advanced temporal smoothing and high-pass filtering to create stable, textured pen-and-ink style effect.

### Key Features

1. **Temporal Smoothing** (lines 52-70, 91-109)
   - Maintains previous frame buffer
   - Alpha blending: 0.3 new frame + 0.7 previous frame
   - Reduces flicker from camera auto-adjustments
   - Also smooths grayscale separately for maximum stability

2. **Brightness Normalization** (lines 111-120)
   - Tracks mean brightness over time
   - Gradually adapts: `current_mean = current_mean * 0.9 + frame_mean * 0.1`
   - Compensates for auto-exposure changes
   - Clamped adjustment (0.8-1.2) prevents dramatic changes

3. **High-Pass Texture Filter** (lines 122-132)
   - Creates low-pass filter with tiny 3×3 Gaussian blur
   - Subtracts from original to get high-frequency detail
   - Amplifies detail by 25× (!)
   - Stretches contrast to full 0-255 range then doubles it

4. **Extreme Texture Enhancement** (line 124)
   - `high_pass * 25.0` - massive amplification
   - Reveals finest details invisible to naked eye
   - Creates textured, sketchy appearance

### Technical Implementation
```python
# Temporal smoothing
if self.prev_frame is not None:
    frame = cv2.addWeighted(frame, 0.3, self.prev_frame, 0.7, 0)

# Brightness normalization
frame_mean = gray.mean()
self.current_mean = self.current_mean * 0.9 + frame_mean * 0.1
adjustment = np.clip(self.target_mean / self.current_mean, 0.8, 1.2)
gray = cv2.convertScaleAbs(gray, alpha=adjustment, beta=0)

# High-pass texture filter
low_pass = cv2.GaussianBlur(gray, (3, 3), 0)
high_pass = gray - low_pass
high_pass = high_pass * 25.0  # Extreme amplification
```

### Unique Features
- **Dual temporal smoothing**: Both RGB and grayscale smoothed separately
- **Adaptive brightness compensation**: Gradually adjusts to lighting changes
- **Extreme high-pass amplification**: 25× boost reveals microscopic texture
- **Double-range contrast stretch**: Expands to 510 range centered at 0

### Use Case
Most stable and detailed pen-and-ink effect. Best for environments with changing lighting conditions.

---

## Branch Lines-Sketch

```bash
git checkout lines-sketch
```

### Effect Description
Sketch-style line rendering for artistic hand-drawn effects.

### Key Features
1. **Sketch Rendering**
   - Hand-drawn appearance
   - Artistic line variation

---

### Refraction/Optical Effects

Optical distortion and refraction-based effects.

---

## Branch Refraction-Cut-Glass

```bash
git checkout refraction-cut-glass
```

### Effect Description
Cut glass refraction effect creating prismatic distortions.

### Key Features
1. **Cut Glass Pattern**
   - Prismatic refraction
   - Angular distortion patterns

---

## Branch Refraction-Rain-Drops

```bash
git checkout refraction-rain-drops
```

### Effect Description
Water droplet refraction effect where drops fall down the screen, refracting the background image like real water on glass.

### Key Features

1. **Pre-Calculated Refraction Maps** (lines 19-103)
   - 5 drop sizes: 8, 10, 12, 15, 18 pixels wide
   - Each size has pre-computed displacement maps
   - Drop height = 2× width (elongated teardrop shape)
   - Maps calculated once in `__init__()` for performance

2. **Realistic Drop Shape** (lines 42-95)
   - **Top half (0-50%)**: Narrow stretched tail
     - Max radius grows from 0.2 to 0.7 of half-width
     - Lower alpha (0.3-0.8) for translucent tail
   - **Middle (50-85%)**: Widening with sine curve
     - Smooth rounding using `np.sin(progress * π/2)`
     - Alpha 0.8-1.0 for more opaque bulb
   - **Bottom (85-100%)**: Smooth point using cosine
     - `max_radius = half_w * cos(progress * π/2)`
     - Creates rounded teardrop point

3. **Physical Refraction Model** (lines 76-95)
   - Strong refraction at edges (like real water drops)
   - **Near edge (<30% from edge)**: Refract strength up to 60.0
   - **Center**: Moderate refraction 15.0
   - Horizontal refraction stronger than vertical
   - Anti-aliasing with wide edge feathering (4 pixel falloff)

4. **Massive Drop Count** (lines 30-31, 125)
   - 1000 active drops simultaneously
   - Creates heavy rain effect
   - Drops respawn immediately when leaving screen

5. **Variable Speed** (lines 111, 119)
   - Speed: 8.0-24.0 pixels/frame
   - Creates depth perception (faster = closer)

6. **Enhanced Source Frame** (lines 130-134)
   - 1.3× saturation boost
   - Subtle contrast boost (1.15× value, -10 offset)
   - Makes refraction more visible

### Technical Implementation
```python
# Pre-calculate refraction for drop shape
for y in range(drop_height):
    for x in range(drop_width):
        # Calculate distance from center
        dist_from_center = np.sqrt(fx * fx)

        # Inside drop - apply refraction
        if dist_from_center < max_radius:
            edge_dist = max_radius - dist_from_center
            strength = edge_dist / max_radius

            # Strong edge refraction
            if strength < 0.3:
                refract_strength = 60.0 * (1.0 - strength / 0.3)
            else:
                refract_strength = 15.0

            offset_x[y, x] = np.sign(dx) * refract_strength
            offset_y[y, x] = refract_strength * 0.3
            alpha[y, x] = edge_falloff * alpha_strength
```

### Unique Features
- **Pre-computed refraction mapping**: Displacement maps calculated once
- **Physically-based lens distortion**: Simulates real water drop optics
- **Shape-based alpha variation**: Translucent tail, opaque bulb
- **Sorted rendering**: Draws top-to-bottom for correct overlapping

### Performance Optimizations
- Pre-calculated maps (not computed per-frame)
- Maps stored in dictionary keyed by size
- Alpha blending allows proper overlap

### Visual Effect
Simulates rain on camera lens or water running down glass. Most realistic optical distortion effect.

---

## Branch Refraction-Square-Lenses

```bash
git checkout refraction-square-lenses
```

### Effect Description
Fisheye lens grid effect dividing the screen into square regions, each with magnified fisheye distortion.

### Key Features

1. **Grid Layout** (lines 19-27)
   - Lens size: 80×80 pixels
   - Grid calculated: `cols = width // 80`, `rows = height // 80`
   - Covers entire screen with tiled lenses

2. **Pre-Calculated Fisheye Mapping** (lines 22-23, 33-56)
   - Single fisheye map computed in `__init__()`
   - Reused for all lenses
   - Strength: 1.5 (moderate distortion)

3. **Barrel Distortion Formula** (lines 39-54)
   - Normalized coordinates from -1 to 1
   - Radial distance: `r = sqrt(nx² + ny²)`
   - Distortion: `r_distorted = r * (1 + strength * r²)`
   - Converts to Cartesian: `x = r * cos(θ)`, `y = r * sin(θ)`

4. **Vectorized Rendering** (lines 33-56)
   - Uses NumPy mgrid for coordinate generation
   - All offsets calculated simultaneously
   - Advanced indexing for bulk pixel grabbing

5. **Border Drawing** (lines 98-102)
   - Dark gray borders (50, 50, 50) around each lens
   - 1 pixel thickness
   - Separates lenses visually

### Technical Implementation
```python
# Create coordinate grids
y_coords, x_coords = np.mgrid[0:size, 0:size]

# Normalized coordinates
nx = (x_coords - half_size) / half_size
ny = (y_coords - half_size) / half_size

# Fisheye distortion
r = np.sqrt(nx * nx + ny * ny)
theta = np.arctan2(ny, nx)
r_distorted = r * (1 + strength * r * r)

# Source coordinates
offset_x = (r_distorted * half_size * np.cos(theta)).astype(np.int32)
offset_y = (r_distorted * half_size * np.sin(theta)).astype(np.int32)

# Apply to region
lens_region = frame[source_y, source_x]
```

### Unique Features
- **Barrel distortion mapping**: Classic fisheye lens mathematics
- **Grid-based tiling**: Screen divided into regular grid
- **Single map reuse**: Same distortion applied to all lenses
- **Vectorized pixel sampling**: NumPy advanced indexing

### Visual Effect
Insect-eye or compound lens effect. Each square shows magnified fisheye view of its region. Creates surreal, fragmented perspective.

### Performance Notes
Very efficient: map calculated once, vectorized operations throughout, no per-pixel loops in rendering.

---

## Common Technical Patterns

### 1. Pre-generation for Consistency
**Pattern**: Generate animated elements in `__init__()` instead of `draw()`
**Benefits**: Consistent appearance, better performance
**Examples**: Christmas ornament balls, fall leaves, winter snowflakes, summer heat waves

### 2. Performance Optimization via Downsampling
**Technique**: Downsample → Process → Upsample
**Speedup**: 9× (33% in each dimension = 1/9 pixels)
**Branches**: Fall, Stained-glass
```python
small = cv2.resize(frame, (w // 3, h // 3))
processed = expensive_operation(small)
result = cv2.resize(processed, (w, h))
```

### 3. Edge Detection Pipeline
**Standard approach**:
1. Convert to grayscale
2. Gaussian blur (5×5)
3. Canny edge detection
4. Optional dilation/blur for thickness
5. Alpha blend onto frame

### 4. Alpha Blending
**Formula**: `result = (A * alpha) + (B * (1 - alpha))`
**Usage**: Smooth compositing of layers
**Examples**: All edge effects, snowflakes, heat waves

### 5. Color Space Transformations
**HSV**: Hue manipulation (summer golden sunset)
**LAB**: Color temperature adjustments
**BGR**: Standard OpenCV format

### 6. Progressive Blur Cycles
**Pattern**: Multiple Gaussian blurs with increasing kernel sizes
**Purpose**: Extend effect reach without blockiness
**Example**: Winter edge mask (12 cycles: 15×15 → 91×91)

### 7. Displacement Mapping
**Pattern**: Precompute offset maps, apply per-pixel distortion
**Benefits**: Realistic refraction, efficient rendering
**Examples**: Summer heat waves, rain-drops, square-lenses

### 8. Bit-Plane Decomposition
**Pattern**: Extract individual bit planes, process separately, recombine
**Technique**: `(channel_data >> bit) & 1` for each bit 0-7
**Benefits**: Reveals data structure, creates layered effects
**Examples**: color-dense-lines (24 layers), mono-lines-from-24-channels (8 layers)

### 9. Temporal Smoothing
**Pattern**: Blend current frame with previous frame buffer
**Formula**: `frame_new = alpha * frame_current + (1 - alpha) * frame_previous`
**Benefits**: Reduces flicker, stabilizes effects in changing lighting
**Example**: mono-traditional-canny-lines (0.3/0.7 blend ratio)

### 10. Physics-Based Animation
**Pattern**: Simulate particle physics with forces, velocities, damping
**Components**: Position, velocity, acceleration, collision detection
**Benefits**: Natural, realistic movement
**Example**: matrix-old-moving-char-physics-version (individual character particles)

### 11. Saturation Manipulation
**Pattern**: Convert to HSV, boost saturation channel, convert back to BGR
**Multiplier**: Typically 2.5× for vivid colors, 1.3× for subtle enhancement
**Examples**: matrix-color (2.5×), rain-drops (1.3×), color-dense-lines (2.5×)

---

## Key Learnings

### 1. Avoid Morphological Operations for Smooth Effects
**Problem**: Dilation creates blocky, rectangular artifacts
**Solution**: Use progressive Gaussian blur cycles instead
**Branch**: Winter

### 2. Pre-compute Expensive Operations
**Examples**:
- Displacement maps (summer heat waves)
- Animated element properties (all branches)
- Color palettes (fall)

### 3. Vectorization Over Loops When Possible
**Preferred**: NumPy array operations
**When loops needed**: Complex per-pixel operations (heat wave refraction)

### 4. User Feedback is Iterative
**Pattern**: Show intermediate results, adjust based on feedback
**Example**: Winter edge mask went through 10+ iterations

### 5. Realistic Shapes Matter
**Problem**: Simple geometric shapes (circles, ovals) look artificial
**Solution**: Use irregular polygons with variation
**Examples**: Snowflakes (3-5 points), leaves (pointed teardrop)

---

## Git Branch Summary

### All Branches (15 Total)

| Branch | Primary Effect | Key Technical Feature |
|--------|---------------|----------------------|
| **Seasonal Effects** |
| christmas | Pine garland + ornaments | Pre-generated reflective balls |
| winter | Arctic blue edges + snow | Progressive Gaussian blur cycles (12 stages) |
| fall | Autumn colors + leaves | Downsampling + color palette mapping |
| summer | Golden sunset + heat waves | HSV hue replacement + thermal refraction |
| **Matrix Variants** |
| matrix-color | Colorful Matrix rain | 2.5× saturation boost + colored character streamers |
| matrix-green | Classic green Matrix | Monochromatic channel isolation |
| matrix-old-moving-char-physics-version | Physics-based Matrix flow | Individual character particle physics |
| **Edge/Line Effects** |
| color-dense-lines | 24-layer bit-plane edges | RGB bit-plane decomposition |
| mono-lines-from-24-channels | Pen-and-ink on white | Grayscale bit-plane + inverted rendering |
| mono-traditional-canny-lines | Stable textured sketch | Temporal smoothing + 25× high-pass filter |
| **Other Effects** |
| stained-glass | K-means quantization | Downsampling optimization |
| rain-drops | Water droplet refraction | Pre-computed displacement maps |
| square-lenses | Compound fisheye grid | Barrel distortion + vectorized rendering |
| **Base** |
| main | White edges on color | Simple Canny edge detection |

### Complexity Ranking

1. **matrix-old-moving-char-physics-version** - Most complex: individual character physics, collision detection, flow simulation
2. **rain-drops** - Complex: pre-computed refraction maps, realistic optical physics
3. **summer** - Complex: thermal heat wave displacement mapping + HSV manipulation
4. **color-dense-lines** - Moderate: 24 bit-plane analysis layers
5. **mono-traditional-canny-lines** - Moderate: temporal smoothing, brightness normalization
6. **winter** - Moderate: 12-stage progressive blur cycles
7. **square-lenses** - Moderate: fisheye mathematics, grid layout
8. **christmas** - Moderate: pre-generated ornaments with reflective rendering
9. **fall** - Simple: color palette mapping with downsampling
10. **matrix-color** - Simple: saturation boost + colored edges
11. **matrix-green** - Simple: monochrome conversion
12. **stained-glass** - Simple: k-means with downsampling
13. **mono-lines-from-24-channels** - Simple: grayscale bit-plane decomposition
14. **main** - Simple: basic edge detection

### Performance Characteristics

**Fastest (20-30 fps):**
- main (simple edge detection)
- matrix-green (green channel mapping)
- matrix-color (HSV conversion + character rendering)

**Moderate (10-20 fps):**
- fall (9× speedup via downsampling)
- stained-glass (9× speedup via downsampling)
- square-lenses (pre-computed maps, vectorized)
- christmas (pre-generated ornaments)
- winter (GPU-accelerated blur cycles)
- summer (pre-computed heat waves)

**Slower (5-15 fps):**
- mono-lines-from-24-channels (8 bit planes)
- color-dense-lines (24 Canny edge detections per frame)
- mono-traditional-canny-lines (temporal smoothing + normalization)

**Slowest (<10 fps):**
- rain-drops (1000 drops with alpha blending)
- matrix-old-moving-char-physics-version (collision detection + particle physics)

---

## Performance Metrics

### Optimization Results
- **Stained-glass**: < 1 fps → ~9 fps (9× speedup via 33% downsampling)
- **Fall**: Similar 9× speedup via downsampling
- **Winter**: Smooth performance despite 12 blur cycles (GPU acceleration)

### Target Frame Rates
- Real-time webcam: 20-30 fps desired
- Acceptable minimum: 10-15 fps

---

## Code Architecture

### Class Structure
```python
class MatrixRain:
    """Effect-specific name (e.g., Heat wave effect)"""

    def __init__(self, width, height):
        # Pre-generate all animated elements
        # Precompute expensive operations
        pass

    def update(self):
        # Update animation state
        # Move elements, check boundaries
        pass

    def draw(self, frame, face_mask=None):
        # Apply visual effects
        # Return processed frame
        pass

class EdgeDetector:
    """Detect edges and faces using MediaPipe"""
    # Used for obstacle detection in some effects
    pass

def main():
    # Camera selection
    # Initialize effect and detector
    # Main loop: read → update → draw → display
    pass
```

### Control Flow
1. Find available cameras
2. User selects camera
3. Initialize `MatrixRain` and `EdgeDetector`
4. Main loop:
   - Capture frame
   - Mirror horizontally
   - If effect enabled: `update()` → `draw()`
   - Display result
   - Handle keyboard input (spacebar toggle, q/ESC quit)

### Keyboard Controls
- **SPACEBAR**: Toggle effect on/off
- **Q or ESC**: Quit
- **Ctrl+C**: Force exit

---

## File Paths

### Repository Location
`/Users/mbennett/Dropbox/dev/webcam-filters/webcam_filter.py`

### Each branch has its own version of `webcam_filter.py`

---

## Dependencies

```python
import cv2                # OpenCV for image processing
import numpy as np        # Array operations
import mediapipe as mp    # Face detection
import random            # Randomization
import time              # FPS calculation
import sys               # System operations
import os                # OS operations
import signal            # Signal handling (Ctrl+C)
```

---

## Future Enhancement Ideas

1. **Performance**: GPU acceleration for heat wave refraction
2. **Spring branch**: Cherry blossoms, rain showers, green tints
3. **Combination effects**: Multiple seasonal elements
4. **User controls**: Sliders for effect intensity, element count
5. **Recording**: Save filtered video to file
6. **Multiple effects**: Blend between seasons
7. **Edge detection optimization**: Pre-compute edges at lower resolution
8. **Heat wave variations**: Different refraction patterns (fire, water, mirage)

---

## Session Highlights

### Most Complex Effects
1. **matrix-old-moving-char-physics-version**: Individual character particle physics with collision detection and fluid dynamics
2. **rain-drops**: Pre-computed refraction maps with physically-based lens distortion
3. **Summer heat waves**: Pre-computed displacement mapping with thermal refraction
4. **Winter edge processing**: 12-stage progressive blur optimization
5. **Christmas ornament balls**: Reflective appearance with multiple highlights

### Most Significant Optimizations
1. **Fall/Stained-glass downsampling**: 9× speedup (process at 33% resolution)
2. **Pre-generation pattern**: Consistent appearance + better performance (ornaments, leaves, snowflakes, heat waves, rain drops, fisheye maps)
3. **Progressive blur cycles**: Smooth effects without blockiness (winter)
4. **Vectorized rendering**: NumPy advanced indexing (square-lenses)
5. **Pre-computed displacement maps**: Calculate once, reuse every frame (rain-drops, square-lenses, summer)

### Most Innovative Techniques
1. **Bit-plane decomposition**: Extract edges from individual bit planes (color-dense-lines: 24 layers, mono-lines-from-24-channels: 8 layers)
2. **Temporal smoothing**: Frame blending to reduce flicker from camera auto-adjustments (mono-traditional-canny-lines)
3. **Extreme high-pass filtering**: 25× amplification reveals microscopic texture (mono-traditional-canny-lines)
4. **Per-character physics**: Each character is independent particle with velocity and collision detection (matrix-old-moving-char-physics-version)
5. **Adaptive brightness normalization**: Gradually compensates for auto-exposure changes (mono-traditional-canny-lines)

### Most Iterations Required
1. **Winter edge mask**: 10+ iterations to achieve smooth, extensive coverage
2. **Summer gradient**: Multiple adjustments to strength and extent
3. **Fall leaf shape**: From circles → ovals → realistic teardrop

### Effect Categories Summary

**Seasonal/Thematic** (4 branches):
- christmas, winter, fall, summer

**Matrix-Inspired** (3 branches):
- matrix-color (colorful modern), matrix-green (classic green), matrix-old-moving-char-physics-version (physics-based)

**Artistic/Sketch** (3 branches):
- color-dense-lines (colorful bit-plane), mono-lines-from-24-channels (pen-and-ink), mono-traditional-canny-lines (stable sketch)

**Optical Effects** (3 branches):
- rain-drops (water refraction), square-lenses (compound fisheye), stained-glass (color quantization)

**Base** (1 branch):
- main (simple white edges)

---

## Documentation Updates

**November 15, 2024** - Comprehensive documentation update:
- Added 9 previously undocumented branches (matrix variants, edge/line effects, optical effects)
- Total branches documented: 15 (up from 6)
- Added complexity rankings and performance characteristics
- Expanded Common Technical Patterns from 7 to 11 patterns
- Enhanced Session Highlights with new categories and innovations
- Added comparative analysis across all branches

---

<<<<<<< HEAD
## Branch: Main - Tkinter Display Implementation
=======
## Branch Main - Tkinter Display Implementation
>>>>>>> 2ddc8e8 (Fix TOC links and remove merge conflict markers)

### Latest Update: November 16, 2024

**Problem**: OpenCV's `cv2.imshow()` on macOS was applying automatic contrast/gamma adjustment that made intermediate gray values (64, 128, 192) appear as pure black or white when displaying grayscale bit planes. The actual pixel values were correct (verified by saving to PNG), but the display rendering was incorrect.

**Solution**: Replaced OpenCV display with Tkinter window using PIL/ImageTk for accurate image rendering without automatic adjustments.

### Implementation Details

1. **Added PIL Import** (line 15)
   ```python
   from PIL import Image, ImageTk
   ```

2. **Created VideoWindow Class** (lines 404-480)
   - Tkinter `Toplevel` window for video display
   - Converts BGR frames to RGB for PIL/ImageTk
   - Handles keyboard input (SPACE, Q, ESC)
   - Position: 420 pixels from left (doesn't overlap control panel)

   Key methods:
   - `update_frame(frame_bgr)`: Converts BGR→RGB→PIL→ImageTk and displays
   - `set_key_callback(callback)`: Sets keyboard handler
   - Keyboard bindings handle spacebar (toggle effect), Q/ESC (quit)

3. **Modified main() Function**
   - Removed `cv2.startWindowThread()`, `cv2.namedWindow()`, `cv2.imshow()`, `cv2.waitKey()`
   - Created `VideoWindow` instance with keyboard callback
   - Replaced `cv2.imshow()` with `video_window.update_frame()`
   - Check `video_window.is_open` instead of OpenCV window property

### Benefits
- **Accurate grayscale display**: No automatic contrast/gamma adjustment
- **Consistent UI**: Both control panel and video window use Tkinter
- **Better cross-platform behavior**: More predictable rendering
- **Correct display of bit-plane values**: Now shows 4 distinct gray levels (0, 64, 128, 192) as expected

### Related Files
- Implementation: `/Users/mbennett/Dropbox/dev/webcam-filters/webcam_filter.py`
- Planning document: `/Users/mbennett/Dropbox/dev/webcam-filters/tkinter_display_plan.md`

---

## Branch Main - Advanced FFT Filtering System (November 17, 2024)

### Latest Major Update: Comprehensive UI Overhaul

The main branch has evolved from simple edge detection into a sophisticated FFT (Fast Fourier Transform) frequency filtering system with multiple output modes and real-time visualization.

### New Features

#### 1. **Output Mode System**
Four distinct output modes with independent control:

**Grayscale Composite** (default)
- Classic single-channel frequency filtering
- Single radius and smoothness control
- Butterworth filter in frequency domain

**Individual Color Channels**
- Per-channel RGB filtering with independent controls
- Each channel (Red, Green, Blue) has:
  - Enable/disable toggle
  - Independent FFT radius (1-350px, exponential scale)
  - Independent smoothness (1-10)
- Allows creative color separation effects

**Grayscale Bit Planes**
- Filters each of 8 bit planes independently
- Bit plane controls (Bit 7 MSB to Bit 0 LSB):
  - Enable/disable per bit
  - Independent radius and smoothness
- Reveals underlying data structure
- Creates multi-layered grayscale effects

**Color Bit Planes**
- Advanced: filters bit planes for each RGB channel
- Total: 24 independent bit planes (8 per channel × 3 channels)
- Tabbed interface (Red/Green/Blue tabs)
- Most computationally intensive mode
- Creates complex colorful layered effects

#### 2. **Filter Curve Visualization Window**
Real-time graph showing frequency response:
- Separate matplotlib window with live updates
- Shows Butterworth filter curves for all active filters
- Updates immediately when mode or parameters change
- X-axis: Distance from FFT center (pixels)
- Y-axis: Mask value (0=blocked, 1=passed)
- Color-coded curves:
  - RGB mode: Red, green, blue lines
  - Grayscale bit planes: Rainbow gradient (purple→red for MSB→LSB)
  - Color bit planes: RGB rainbow gradient
- Fixed to prevent stale visualization when switching modes

#### 3. **Collapsible Control Panels**
Organized UI with expand/collapse functionality:
- Camera Controls (always visible)
- Output Mode Selection (always visible)
- Individual Color Channels (collapsible)
- Grayscale Bit Planes (collapsible with 8-row table)
- Color Bit Planes (collapsible with tabbed interface)

#### 4. **Settings Persistence**
All settings saved to `~/.webcam_filter_settings.json`:
- Camera selection
- Output mode
- All filter parameters (radius, smoothness, enable states)
- UI state (expanded/collapsed panels)
- Mirror flip preference
- Restored on application restart

#### 5. **Mirror Flip Option**
Toggle between Normal View and Flip Left/Right:
- Radio buttons in Camera Controls
- Persisted in settings
- Loaded on startup
- Applied conditionally in video processing

### Technical Implementation

**FFT Frequency Filtering Pipeline** (lines 374-852):
1. Convert to grayscale or split RGB channels
2. Apply FFT: `cv2.dft(float_data, flags=cv2.DFT_COMPLEX_OUTPUT)`
3. Shift zero-frequency to center: `np.fft.fftshift()`
4. Create Butterworth filter mask based on radius/smoothness
5. Apply mask to frequency domain
6. Inverse FFT: `cv2.idft()` after `ifftshift()`
7. Reconstruct image from filtered data

**Butterworth Filter** (lines 305-373):
```python
def create_butterworth_mask(rows, cols, radius, smoothness=2):
    center_y, center_x = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Butterworth formula: H(u,v) = 1 / (1 + (D/D0)^(2n))
    if radius > 0:
        mask = 1.0 / (1.0 + np.power(distance / radius, 2 * smoothness))
    else:
        mask = np.zeros((rows, cols), np.float32)

    return np.stack([mask, mask], axis=-1)  # Complex mask
```

**Bit Plane Extraction** (lines 633-761):
```python
# Extract individual bit planes
for bit in range(8):
    bit_plane = ((gray_frame >> bit) & 1).astype(np.float32) * 255.0
    # Apply FFT filtering to this bit plane
    filtered_bit_plane = apply_fft_filter(bit_plane, params)
    # Reconstruct with threshold
    reconstructed_bit = (filtered_bit_plane > 127.5).astype(np.uint8)
    result += reconstructed_bit << bit
```

**Visualization Refresh** (lines 2088-2153):
- New `_refresh_visualization()` method
- Called by radio button callbacks
- Immediately updates filter curve graph when mode changes
- Gathers current parameters based on active mode
- Creates dummy distance array for visualization
- Triggers matplotlib canvas redraw

### UI Components

**Control Panel** (Tkinter):
- Width: 400 pixels
- Collapsible sections with ▶/▼ indicators
- Tables for bit plane controls:
  - Grayscale: 8 rows (Bit 7-0)
  - Color: 3 tabs × 8 rows each
- Exponential sliders for radius (1-350px)
- Linear sliders for smoothness (1-10)
- Checkboxes for enable/disable
- Mirror flip radio buttons

**Video Display** (Tkinter + PIL):
- Replaced OpenCV's `cv2.imshow()` with Tkinter window
- Accurate grayscale rendering (no auto-contrast)
- Position: 420px from left (doesn't overlap control panel)
- Handles keyboard input (SPACE, Q, ESC)

**Filter Visualization** (Matplotlib):
- Embedded FigureCanvasTkAgg
- Real-time filter curve updates
- Grid with reference lines at 0, 0.03, 0.5, 1.0
- Legend with filter descriptions
- Title changes based on mode

### Recent Bug Fixes

**November 17, 2024 - Visualization Mode Switching Issue**:
- **Problem**: When switching output modes (e.g., Color Bit Planes → Grayscale Bit Planes), the filter curve graph showed stale data from the previous mode
- **Root Cause**: Visualization only updated during video frame processing, not immediately when radio buttons were clicked
- **Solution**: Added `_refresh_visualization()` method that:
  1. Reads current output mode
  2. Gathers appropriate parameters (RGB, grayscale bitplane, or color bitplane)
  3. Calls `update_visualization()` with correct params
  4. Forces matplotlib canvas redraw
- **Implementation**: Added calls to `_refresh_visualization()` in all three radio button callbacks:
  - `_on_rgb_radio_select()`
  - `_on_bitplane_radio_select()`
  - `_on_color_bitplane_radio_select()`

### Performance Characteristics

**Complexity by Mode**:
1. **Grayscale Composite**: Fastest (1 FFT + 1 IFFT)
2. **Individual Color Channels**: Moderate (3 FFTs, one per channel)
3. **Grayscale Bit Planes**: Slower (8 FFTs, one per bit plane)
4. **Color Bit Planes**: Slowest (24 FFTs, 8 per channel × 3)

**Optimizations**:
- Pre-computed Butterworth masks cached when parameters don't change
- Visualization updates only when needed (not every frame)
- Settings loaded/saved only on startup/change
- Collapsible UI reduces widget overhead

### File Locations
- Main implementation: `/Users/mbennett/Dropbox/dev/webcam-filters/webcam_filter.py`
- Settings file: `~/.webcam_filter_settings.json`
- Planning docs: `/Users/mbennett/Dropbox/dev/webcam-filters/tkinter_display_plan.md`

### Dependencies
- **cv2** (OpenCV): FFT operations, image processing
- **numpy**: Array operations, FFT shifting
- **tkinter**: UI framework for controls and video
- **matplotlib**: Filter curve visualization
- **PIL** (Pillow): Image conversion for Tkinter display
- **json**: Settings persistence

---

*Last updated: November 17, 2024*
*Repository: webcam-filters*
*Primary file: webcam_filter.py (branch-specific versions)*
*Total branches: 21*
