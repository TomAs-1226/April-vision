#pragma once

#ifdef config
#undef config
#endif

#include <cstdint>

namespace config {

// Display settings
inline constexpr int MAX_DISPLAY_W = 1200;
inline constexpr int MAX_DISPLAY_H = 800;

// Tag parameters
inline constexpr double DEFAULT_TAG_SIZE_M = 0.1524;

// Camera settings
inline constexpr int CAM_IDX = 0;
inline constexpr int CAPTURE_FPS = 60;
inline constexpr int FRAME_UI_MS = 12;

// Performance tuning
inline constexpr int DETECTION_RATE_HZ = 60;
inline constexpr int GUI_RATE_HZ = 20;
inline constexpr int MIN_DET_RATE = 8;
inline constexpr int MAX_DET_RATE = 120;

// NetworkTables / UDP
inline constexpr const char* NT_SERVER = "10.0.0.2";
inline constexpr bool UDP_FALLBACK = true;
inline constexpr const char* UDP_TARGET_IP = "10.0.0.11";
inline constexpr int UDP_TARGET_PORT = 5800;

// Detection parameters
inline constexpr int DEFAULT_DECIMATE = 1;
inline constexpr double DEFAULT_CONF = 0.18;
inline constexpr double REPROJ_ERR_THRESH = 2.2;

// Motion/adaptive decimation
inline constexpr double BLUR_HIGH = 30.0;
inline constexpr double BLUR_MED = 80.0;
inline constexpr int ADAPT_DECIMATE_HIGH = 3;
inline constexpr int ADAPT_DECIMATE_MED = 2;
inline constexpr int ADAPT_DECIMATE_LOW = 1;

// Per-tag timeouts
inline constexpr double KEEPALIVE_TIMEOUT = 0.4;
inline constexpr double REMOVAL_TIMEOUT = 0.8;
inline constexpr double SCENE_PURGE_TIMEOUT = 1.2;

// Dynamics
inline constexpr double VELOCITY_DECAY = 0.82;
inline constexpr double MAX_PREDICT_DISTANCE = 0.45;
inline constexpr double MIN_SCALE_PX = 6.0;
inline constexpr double MAX_SCALE_PX = 5000.0;

// Optical flow
inline constexpr int LK_WIN_SIZE = 21;
inline constexpr int LK_MAX_LEVEL = 3;
inline constexpr int LK_MAX_ITER = 20;
inline constexpr double LK_EPSILON = 0.03;

// Kalman defaults
inline constexpr double TRACK_Q = 0.02;
inline constexpr double TRACK_R = 4.0;

// EMA smoothing defaults
inline constexpr double EMA_ALPHA_POS = 0.28;
inline constexpr double EMA_ALPHA_POSE = 0.22;

// Pose median smoothing window
inline constexpr int POSE_MEDIAN_WINDOW = 5;

// Safety/CPU thresholds
inline constexpr double PROCESS_TIME_HIGH_MS = 18.0;
inline constexpr double PROCESS_TIME_LOW_MS = 8.0;

// CLAHE parameters
inline constexpr double CLAHE_CLIP_LIMIT = 2.2;
inline constexpr int CLAHE_TILE_SIZE = 8;

// Corner refinement
inline constexpr int CORNER_SUBPIX_WIN = 5;
inline constexpr int CORNER_SUBPIX_MAX_ITER = 10;
inline constexpr double CORNER_SUBPIX_EPS = 0.03;

} // namespace config
