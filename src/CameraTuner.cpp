#include "CameraTuner.h"
#include <sstream>

using std::string;
using std::vector;
using std::pair;

namespace camtuner {

// Properties we might set/read
static const int P_EXPOSURE    = cv::CAP_PROP_EXPOSURE;
static const int P_AUTOEXP     = cv::CAP_PROP_AUTO_EXPOSURE;
static const int P_GAIN        = cv::CAP_PROP_GAIN;
static const int P_BRIGHT      = cv::CAP_PROP_BRIGHTNESS;
static const int P_CONTRAST    = cv::CAP_PROP_CONTRAST;
static const int P_SAT         = cv::CAP_PROP_SATURATION;
static const int P_GAMMA       = cv::CAP_PROP_GAMMA;
static const int P_FPS         = cv::CAP_PROP_FPS;
static const int P_W           = cv::CAP_PROP_FRAME_WIDTH;
static const int P_H           = cv::CAP_PROP_FRAME_HEIGHT;

static void pushLog(vector<string>& log, const string& k, double v) {
    std::ostringstream oss; oss << k << "=" << v; log.push_back(oss.str());
}

vector<pair<int,double>> readAllKnownProps(cv::VideoCapture& cap) {
    vector<pair<int,double>> out = {
        {P_W,0},{P_H,0},{P_FPS,0},{P_AUTOEXP,0},{P_EXPOSURE,0},
        {P_GAIN,0},{P_BRIGHT,0},{P_CONTRAST,0},{P_SAT,0},{P_GAMMA,0}
    };
    for (auto& kv : out) kv.second = cap.get(kv.first);
    return out;
}

static void setIfSupported(cv::VideoCapture& cap, int prop, double val) {
    if (val < -0.5) return; // -1 means skip
    cap.set(prop, val);
}

static void warmup(cv::VideoCapture& cap, int n) {
    cv::Mat f;
    for (int i=0;i<n;i++) cap.read(f);
}

OpenResult openAndTune(const Settings& s) {
    OpenResult r;
    for (int api : s.backendOrder) {
        r.cap.release();
        r.cap.open(s.index, api);
        if (!r.cap.isOpened()) continue;

        // Base format / FPS first
        r.cap.set(P_W, s.width);
        r.cap.set(P_H, s.height);
        if (s.fps > 0) r.cap.set(P_FPS, s.fps);

        // --- Auto Exposure control semantics (Windows) ---
        // Many UVC drivers use:
        //   0.75 => AE ON,  0.25 => AE OFF (manual exposure)
        // Then CAP_PROP_EXPOSURE uses log2(seconds), e.g. -6 ~ 1/64 s.  :contentReference[oaicite:3]{index=3}
        if (s.useAutoExposure) {
            r.cap.set(P_AUTOEXP, 0.75);  // AE on
        } else {
            r.cap.set(P_AUTOEXP, 0.25);  // AE off (manual)
            r.cap.set(P_EXPOSURE, s.manualExposure);
        }

        setIfSupported(r.cap, P_GAIN, s.gain);
        setIfSupported(r.cap, P_BRIGHT, s.brightness);
        setIfSupported(r.cap, P_GAMMA, s.gamma);

        // Let AE/AGC settle (or sensor warm up) before we judge brightness.
        warmup(r.cap, s.warmupFrames);

        // Verify
        auto props = readAllKnownProps(r.cap);
        for (auto& kv : props) pushLog(r.log, std::to_string(kv.first), kv.second);

        // Do a quick probe frame; reject obviously dark if AE is on and MSMF misbehaved.
        cv::Mat probe;
        if (!r.cap.read(probe) || probe.empty()) { r.ok=false; continue; }

        // If median is very dark, try flipping AE mode and/or backend
        cv::Scalar m = cv::mean(probe);
        double meanGray = (probe.channels()==1) ? m[0] : (0.114*m[0] + 0.587*m[1] + 0.299*m[2]);
        if (meanGray < 15.0) {
            // Try toggling AE state once
            r.cap.set(P_AUTOEXP, s.useAutoExposure ? 0.25 : 0.75);
            if (!s.useAutoExposure) r.cap.set(P_EXPOSURE, s.manualExposure);
            warmup(r.cap, 6);
            r.cap.read(probe);
            m = cv::mean(probe);
            meanGray = (probe.channels()==1) ? m[0] : (0.114*m[0] + 0.587*m[1] + 0.299*m[2]);
        }

        // Accept this backend if we have a sane image or user wanted manual exposure
        r.ok = (meanGray >= 15.0) || (!s.useAutoExposure);
        if (r.ok) return r;
        // otherwise try next backend
    }
    r.ok = false;
    return r;
}

} // namespace camtuner
