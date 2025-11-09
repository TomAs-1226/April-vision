#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <utility>
#include <sstream>

namespace camtuner {

struct OpenResult {
    cv::VideoCapture cap;
    std::vector<std::string> log;
    bool ok = false;
};

struct Settings {
    int index = 0;
    int width = 1280;
    int height = 720;
    double fps = 30.0;
    bool useAutoExposure = true;   // 0.75 = on, 0.25 = off (Windows UVC)
    double manualExposure = -6.0;  // log2(seconds) on Windows
    double gain = -1.0;
    double brightness = -1.0;
    double gamma = -1.0;
    std::vector<int> backendOrder = { cv::CAP_DSHOW, cv::CAP_MSMF };
    int warmupFrames = 12;
};

inline std::vector<std::pair<int,double>> readAllKnownProps(cv::VideoCapture& cap) {
    std::vector<std::pair<int,double>> out = {
        {cv::CAP_PROP_FRAME_WIDTH,0},{cv::CAP_PROP_FRAME_HEIGHT,0},{cv::CAP_PROP_FPS,0},
        {cv::CAP_PROP_AUTO_EXPOSURE,0},{cv::CAP_PROP_EXPOSURE,0},{cv::CAP_PROP_GAIN,0},
        {cv::CAP_PROP_BRIGHTNESS,0},{cv::CAP_PROP_CONTRAST,0},{cv::CAP_PROP_SATURATION,0},
        {cv::CAP_PROP_GAMMA,0}
    };
    for (auto& kv : out) kv.second = cap.get(kv.first);
    return out;
}

inline void warmup(cv::VideoCapture& cap, int n) {
    cv::Mat f;
    for (int i=0;i<n;i++) cap.read(f);
}

inline OpenResult openAndTune(const Settings& s) {
    OpenResult r;
    for (int api : s.backendOrder) {
        r.cap.release();
        r.cap.open(s.index, api);
        if (!r.cap.isOpened()) continue;

        r.cap.set(cv::CAP_PROP_FRAME_WIDTH,  s.width);
        r.cap.set(cv::CAP_PROP_FRAME_HEIGHT, s.height);
        if (s.fps > 0) r.cap.set(cv::CAP_PROP_FPS, s.fps);

        // Windows UVC: 0.75 = AE ON, 0.25 = AE OFF (manual); exposure is log2(seconds).
        // (Driver-dependent, but widely observed.) :contentReference[oaicite:2]{index=2}
        r.cap.set(cv::CAP_PROP_AUTO_EXPOSURE, s.useAutoExposure ? 0.75 : 0.25);
        if (!s.useAutoExposure) r.cap.set(cv::CAP_PROP_EXPOSURE, s.manualExposure);
        if (s.gain       >= 0) r.cap.set(cv::CAP_PROP_GAIN, s.gain);
        if (s.brightness >= 0) r.cap.set(cv::CAP_PROP_BRIGHTNESS, s.brightness);
        if (s.gamma      >= 0) r.cap.set(cv::CAP_PROP_GAMMA, s.gamma);

        warmup(r.cap, s.warmupFrames);

        auto props = readAllKnownProps(r.cap);
        for (auto& kv : props) { std::ostringstream oss; oss << kv.first << "=" << kv.second; r.log.push_back(oss.str()); }

        cv::Mat probe;
        if (!r.cap.read(probe) || probe.empty()) { r.ok = false; continue; }
        cv::Scalar m = cv::mean(probe);
        double meanGray = (probe.channels()==1) ? m[0] : (0.114*m[0] + 0.587*m[1] + 0.299*m[2]);

        if (meanGray < 15.0) { // try flipping AE once
            r.cap.set(cv::CAP_PROP_AUTO_EXPOSURE, s.useAutoExposure ? 0.25 : 0.75);
            if (!s.useAutoExposure) r.cap.set(cv::CAP_PROP_EXPOSURE, s.manualExposure);
            warmup(r.cap, 6);
            r.cap.read(probe);
            m = cv::mean(probe);
            meanGray = (probe.channels()==1) ? m[0] : (0.114*m[0] + 0.587*m[1] + 0.299*m[2]);
        }

        r.ok = (meanGray >= 15.0) || (!s.useAutoExposure);
        if (r.ok) return r;
    }
    r.ok = false;
    return r;
}

} // namespace camtuner
