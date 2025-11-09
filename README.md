# April Vision (Windows Test Build)

**Status:** Windows test build (CLion/Visual Studio + CMake)  
**Repo intent:** Fast AprilTag-based vision pipeline for FRC-style robotics, with pose estimation, lightweight tracking, and UDP publishing.

> This folder is a _non‑latest_ snapshot used for bringing up the Windows build and validating the processing pipeline. It focuses on reliable detection and a clean networking path rather than packaging or cross‑platform polish.

---

## Why I built this

- **Field‑ready perception for FRC**: I wanted a small, self‑contained vision app that can lock onto **AprilTag 36h11** targets, estimate pose, and publish data to the robot with **minimal latency**.
- **Windows bring‑up & debugging**: A Windows test build lets me iterate quickly (CLion/VS debugger, profiler, camera sanity checks) before pushing to the on‑bot Linux SBC.
- **Control‑friendly outputs**: The app emits exactly the values a robot controller expects (ID, corner pixels, filtered center/heading, and 3D pose) over a **simple UDP** protocol. No monolithic frameworks required.

---

## High‑level architecture

```
Camera -> FrameProcessor -> Detector (AprilTag) -> PoseEstimator (solvePnP)
                                     |                         |
                                     +--> Tracker (Kalman) <---+
                                                   |
                                            NetworkPublisher (UDP)
```

### Key components (from `src/`)

- **`Detector.*`**
  - Uses `apriltag` (tag36h11) + OpenCV for grayscale/CLAHE pre‑processing and **corner sub‑pixel refinement**.
  - Exposes a `Detection` struct: `id`, `corners (4x Point2f)`, and derived center.
  - Tunables in `Config.h` (CLAHE clip/tile, subpix window/iters/eps, timing thresholds).

- **`PoseEstimator.*`**
  - Computes camera‑to‑tag pose (rvec/tvec) via OpenCV **`solvePnP`**, given known tag size and intrinsics.
  - Returns translation (meters) and rotation (radians / yaw convenience).

- **`Tracker.*`**
  - Lightweight **Kalman‑style** smoothing on tag center/scale/heading to reduce jitter and provide a stable target when short occlusions happen.
  - Tracks “seconds unseen” to drop stale targets cleanly.

- **`FrameProcessor.*`**
  - Glue that runs detection + pose + tracking per frame, enforces frame timing, and shares latest result with the app/UI thread.
  - Handles corner‑cases near image borders to avoid sub‑pixel failures.

- **`NetworkPublisher.*`**
  - **Windows Winsock UDP** publisher (portable to BSD sockets) that sends the processed packet (ID, center, yaw, pose, timestamp) at camera frame‑rate.
  - Simple queue with a background thread so publishing never stalls the pipeline.

- **`main.cpp` (in `src/`)**
  - Spins up camera capture + processing threads, shows minimal diagnostics, and publishes results.
  - There’s also a CLion template `main.cpp` at repo root; the runtime entrypoint is the one under `src/`.

- **`Config.h`**
  - Centralizes camera index, resolution, tag size (meters), UDP IP/port, and all vision constants (CLAHE, subpix, thresholds).

---

## Features

- **AprilTag 36h11** detection via the C apriltag library integrated with OpenCV.
- **Robust pre‑processing** (CLAHE) and **corner sub‑pixel refinement**.
- **Pose estimation** using camera intrinsics + physical tag size.
- **Kalman‑style tracking** for smoother center/heading and brief occlusions.
- **UDP publishing** of compact target data to the robot (Winsock on Windows).
- **Config‑first** design: tweak parameters in one place (`Config.h`).

> Heads‑up: This is the **Windows test build**; packaging, cross‑platform scripts, and richer CLI flags will come in a later revision.

---

## Getting started (Windows 10/11, x64)

### 1) Prerequisites

- **Visual Studio 2022** (Desktop development with C++) or **Build Tools**  
- **CMake ≥ 3.24**
- **vcpkg** (recommended) with:
  - `opencv[core,video,highgui]`
  - `apriltag` (or provide headers/libs in `thirdparty`)
  - `eigen3`

Install via vcpkg (PowerShell):

```powershell
# Clone vcpkg once (if you don't have it)
git clone https://github.com/microsoft/vcpkg C:
cpkg
C:
cpkgootstrap-vcpkg.bat

# Install dependencies (x64)
C:
cpkg
cpkg.exe install opencv[core,video,highgui]:x64-windows eigen3:x64-windows apriltag:x64-windows

# (Optional) integrate with your VS/CMake
C:
cpkg
cpkg.exe integrate install
```

### 2) Configure & build (CMake)

```powershell
# From repo root
mkdir build
cd build

# If you use vcpkg toolchain:
cmake -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  ..

# Build (Debug example)
cmake --build . --config Debug
```

> The `CMakeLists.txt` attempts to copy needed OpenCV/AprilTag DLLs beside the executable after build when it can find them.

### 3) Run

- Plug in a UVC‑compatible camera.  
- Launch the built exe (in `build/Debug/` or from CLion).  
- Default settings (camera index, UDP target, tag size) come from **`Config.h`**.

---

## Configuration

Edit **`src/Config.h`** to match your setup:

```cpp
namespace config {
    inline constexpr int CAMERA_INDEX = 0;
    inline constexpr int FRAME_WIDTH = 1280;
    inline constexpr int FRAME_HEIGHT = 720;

    inline constexpr double TAG_SIZE_METERS = 0.165; // set your physical tag size

    inline constexpr const char* UDP_IP = "10.58.05.2";  // roboRIO/PI/driver station
    inline constexpr int UDP_PORT = 5805;

    // Vision tuning (CLAHE, sub-pixel, thresholds)...
}
```

If you use a custom camera calibration, place your intrinsics (fx, fy, cx, cy) and distortion in the `PoseEstimator` setup or add a small loader (JSON/YAML).

---

## Data format (UDP)

A compact ASCII/CSV or binary packet is sent per frame (see `NetworkPublisher.*`). A typical payload includes:

- `tag_id`
- `center_px_x, center_px_y`
- `yaw_rad` (or heading)
- `tx, ty, tz` (meters, camera → tag)
- `timestamp`

Adapt the struct/serializer for your robot code as needed.

---

## Common pitfalls & tips

- **`cv::cornerSubPix` assertion** near image borders: if a tag touches the edge, sub‑pixel refinement can sample outside the frame. The code clamps/guards, but you can lower `CORNER_SUBPIX_WIN` or skip sub‑pix when within a small margin.
- **DLLs not found** on launch: make sure OpenCV/AprilTag DLLs are copied next to the exe (the `CMakeLists.txt` has a post‑build copy step for common vcpkg layouts).
- **Latency**: prefer 1280×720@30 over 1920×1080@30 if USB bandwidth is tight; tracking will keep targets smooth.
- **Networking**: verify Windows firewall allows outbound UDP on your chosen port.

---

## Roadmap (post Windows test)

- Cross‑platform runner (Linux ARM64 on Pi/Orin/OrangePi 5).  
- Configurable CLI flags and JSON logging.  
- Multi‑tag selection and 3D field‑relative transforms.  
- NetworkTables/NT4 publisher option.

---

## License

MIT (or project default). See `LICENSE` if present.

---

## Credits

- **AprilTag** (original C implementation).  
- **OpenCV** for image processing & `solvePnP`.  
- **Eigen** for lightweight filtering/linear algebra.
