#pragma once

// Include Windows headers FIRST before anything else
#ifdef _WIN32
#define NOMINMAX  // Prevent Windows from defining min/max macros
#include <windows.h>
#endif

// Now undefine any remaining macros
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <string>
#include <cstdint>

namespace vision_config {

// Display settings
constexpr int MAX_DISPLAY_W = 1200;
constexpr int MAX_DISPLAY_H = 800;

// Tag parameters
constexpr double DEFAULT_TAG_SIZE_M = 0.1524;

// Camera settings
constexpr int CAM_IDX = 0;
constexpr int CAPTURE_FPS = 60;
constexpr int FRAME_UI_MS = 12;

// Performance tuning
constexpr int DETECTION_RATE_HZ = 60;
constexpr int GUI_RATE_HZ = 20;
constexpr int MIN_DET_RATE = 8;
constexpr int MAX_DET_RATE = 120;

// NetworkTables / UDP
const std::string NT_SERVER = "10.0.0.2";
constexpr bool UDP_FALLBACK = true;
const std::string UDP_TARGET_IP = "10.0.0.11";
constexpr int UDP_TARGET_PORT = 5800;

// Detection parameters
constexpr int DEFAULT_DECIMATE = 1;
constexpr double DEFAULT_CONF = 0.18;
constexpr double REPROJ_ERR_THRESH = 2.2;

// Motion/adaptive decimation
constexpr double BLUR_HIGH = 30.0;
constexpr double BLUR_MED = 80.0;
constexpr int ADAPT_DECIMATE_HIGH = 3;
constexpr int ADAPT_DECIMATE_MED = 2;
constexpr int ADAPT_DECIMATE_LOW = 1;

// Per-tag timeouts
constexpr double KEEPALIVE_TIMEOUT = 0.4;
constexpr double REMOVAL_TIMEOUT = 0.8;
constexpr double SCENE_PURGE_TIMEOUT = 1.2;

// Dynamics
constexpr double VELOCITY_DECAY = 0.82;
constexpr double MAX_PREDICT_DISTANCE = 0.45;
constexpr double MIN_SCALE_PX = 6.0;
constexpr double MAX_SCALE_PX = 5000.0;

// Optical flow
constexpr int LK_WIN_SIZE = 21;
constexpr int LK_MAX_LEVEL = 3;
constexpr int LK_MAX_ITER = 20;
constexpr double LK_EPSILON = 0.03;

// Kalman defaults
constexpr double TRACK_Q = 0.02;
constexpr double TRACK_R = 4.0;

// EMA smoothing defaults
constexpr double EMA_ALPHA_POS = 0.28;
constexpr double EMA_ALPHA_POSE = 0.22;

// Pose median smoothing window
constexpr int POSE_MEDIAN_WINDOW = 5;

// Safety/CPU thresholds
constexpr double PROCESS_TIME_HIGH_MS = 18.0;
constexpr double PROCESS_TIME_LOW_MS = 8.0;

// CLAHE parameters
constexpr double CLAHE_CLIP_LIMIT = 2.2;
constexpr int CLAHE_TILE_SIZE = 8;

// Corner refinement
constexpr int CORNER_SUBPIX_WIN = 5;
constexpr int CORNER_SUBPIX_MAX_ITER = 10;
constexpr double CORNER_SUBPIX_EPS = 0.03;

} // namespace vision_config