# 🎯 Arrow Maze Auto-Solver & Infinite Runner

An automated, high-performance bot for mobile Arrow Maze puzzle games on Android. It uses computer vision, in-memory topological raycast simulations, and native Android multi-touch injection via ADB to achieve **100% one-shot board clears** and **continuous multi-level looping**.

---

## 📂 Project Structure

```text
ArrowGame/
├── 📄 loop_play.py           # 🚀 Main entry point for continuous multi-level auto-play
├── 📄 solver_one_shot.py     # 🧠 Core 100% in-memory multi-wave solving engine
├── 📦 pinch.jar              # ⚡ Native Android DEX multi-touch injector for pinch zoom
├── 🖼️ arrow_up_template.png  # 🔍 Base template matching arrowhead
├── 🖼️ btn_next.png           # 🏆 Victory screen 'Continue' button template
│
├── 📁 src/                   # ☕ Java source for native Android multi-touch
│   └── com/touch/PinchZoom.java
│
├── 📁 tools/                 # 🛠️ Calibration, cropping & diagnostics utilities
│   ├── calibration.py        # Correlation score and scale calibration
│   ├── crop_template.py      # Interactive ROI cropper for arrow templates
│   ├── crop_cont.btn.py      # Interactive ROI cropper for victory buttons
│   ├── screenshot.py         # Capture high-res screenshots from device
│   ├── phone_connection.py   # ADB connection test & binarization check
│   └── debug.py              # Visual raycast corridor debugger
│
├── 📁 legacy/                # 📜 Historical solver iterations (solver, solver_pro, solver_pro_max)
└── 📁 debug_images/          # 📸 Sample frames, binary debug masks & victory captures
```

---

## 🌟 Key Features & How It Works

### 1. 100% In-Memory One-Shot Solver (`solver_one_shot.py`)
* **Multi-Scale Head Detection**: Matches arrowheads scaled from $75\%$ to $125\%$ across all 4 cardinal rotations (`UP`, `DOWN`, `LEFT`, `RIGHT`) with Non-Maximum Suppression (NMS).
* **Connected Component Body Erasure**: Maps each detected arrowhead to its full connected stroke mask on the binary image. When an arrow clears, its **entire body (including $90^\circ$ corners and bends)** is erased in memory.
* **Raycast Simulation**: Projects flight rays forward along the arrow direction. If unobstructed, the arrow is scheduled into the current wave.
* **Rapid Discrete Waves**: Solves up to 60 cascading waves in $<50\text{ms}$ in CPU memory, then blasts the entire tap sequence to the phone via a persistent `adb shell` pipe with zero process-spawn overhead ($6\text{–}8\text{ moves/sec}$).
* **Fallback Protection**: If complex overlapping geometry leaves any arrow uncalculated, it automatically takes a fallback screenshot to finish off any remaining arrows.

### 2. Continuous Infinite Runner (`loop_play.py`)
* **Automated Level Progression**: Solves the level $\to$ waits for victory popup $\to$ clicks Continue $\to$ waits for spawn $\to$ zooms out $\to$ loops indefinitely.
* **Spawn Animation Polling**: Guards against premature solver execution while arrows are still in their drop-in animation.
* **Dual-State Transition**: Detects both the victory "Continue" button and auto-advancing levels.

### 3. Native Multi-Touch Pinch Zoom (`pinch.jar`)
* **True Simultaneous 2-Finger Pinch**: Direct binder calls to Android's `IInputManager` via `app_process` inject synchronized `MotionEvent` packets (`pointerCount = 2`).
* **Zero Subprocess Lag**: Eliminates asynchronous CLI spawn delays and touch drift on non-rooted devices.

---

## 🚀 Quick Start Guide

### Prerequisites
1. **Python 3.8+** with dependencies:
   ```powershell
   pip install opencv-python numpy
   ```
2. **Android Phone** with **USB Debugging** enabled and connected via ADB:
   ```powershell
   adb devices
   ```

---

### Running the Auto-Player

#### Option 1: Continuous Multi-Level Loop (Recommended)
Automatically plays level after level unattended:
```powershell
python .\loop_play.py
```

#### Option 2: Single Level Solver
Solves only the current level on screen in one shot:
```powershell
python .\solver_one_shot.py
```

---

## ⚙️ Configuration & Speed Tuning

Both `loop_play.py` and `solver_one_shot.py` accept delay parameters:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `tap_delay` | `0.04s` | Interval between taps within the same wave |
| `wave_delay` | `0.08s` | Settle time between consecutive waves |
| `max_levels` | `100` | Number of levels to play in `loop_play.py` |
| `duration_ms`| `350ms` | Duration of the native pinch zoom gesture |

---

## 🛠️ Utility Scripts (`tools/`)

* **Capture Screen**:
  ```powershell
  python .\tools\screenshot.py
  ```
* **Crop New Arrowhead Template**:
  ```powershell
  python .\tools\crop_template.py
  ```
* **Crop New Continue Button Template**:
  ```powershell
  python .\tools\crop_cont.btn.py
  ```
* **Inspect Corridors & Collision Debugger**:
  ```powershell
  python .\tools\debug.py
  ```
