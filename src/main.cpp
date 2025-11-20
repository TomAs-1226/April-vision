
#include "FrameProcessor.h"
#include "NetworkPublisher.h"
#include "Config.h"
#include "CameraTuner.h"

#include "httplib.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "iphlpapi.lib")
#else
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif

namespace {
class LatestFrameExchange {
public:
    void publish(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        frame.copyTo(buffer_);
        hasFrame_ = true;
        cond_.notify_one();
    }

    bool acquire(cv::Mat& out, std::chrono::milliseconds wait) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, wait, [&] { return !alive_ || hasFrame_; })) {
            return false;
        }
        if (!alive_ && !hasFrame_) {
            return false;
        }
        if (!hasFrame_) {
            return false;
        }
        buffer_.copyTo(out);
        hasFrame_ = false;
        return true;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            alive_ = false;
        }
        cond_.notify_all();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        hasFrame_ = false;
        alive_ = true;
        buffer_.release();
    }

private:
    cv::Mat buffer_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool hasFrame_{false};
    bool alive_{true};
};

struct CameraPropertyRange {
    double min;
    double max;
};

inline double sliderToPropertyValue(int slider, const CameraPropertyRange& range) {
    const double normalized = std::clamp(slider, 0, static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE)) /
                              config::CAMERA_PROP_SLIDER_SCALE;
    return range.min + normalized * (range.max - range.min);
}

std::vector<std::string> localIpAddresses() {
    std::vector<std::string> addrs;
#ifdef _WIN32
    ULONG bufLen = 15000;
    std::vector<unsigned char> buffer(bufLen);
    IP_ADAPTER_ADDRESSES* addresses = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data());
    if (GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                                           GAA_FLAG_SKIP_DNS_SERVER | GAA_FLAG_SKIP_FRIENDLY_NAME,
                              nullptr, addresses, &bufLen) == NO_ERROR) {
        for (auto* addr = addresses; addr != nullptr; addr = addr->Next) {
            if (addr->OperStatus != IfOperStatusUp) continue;
            for (auto* unicast = addr->FirstUnicastAddress; unicast; unicast = unicast->Next) {
                char bufferA[INET_ADDRSTRLEN]{};
                auto* sa = reinterpret_cast<sockaddr_in*>(unicast->Address.lpSockaddr);
                if (sa->sin_family == AF_INET && sa->sin_addr.S_un.S_addr != htonl(INADDR_LOOPBACK)) {
                    inet_ntop(AF_INET, &sa->sin_addr, bufferA, sizeof(bufferA));
                    addrs.emplace_back(bufferA);
                }
            }
        }
    }
#else
    ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) return addrs;
    for (auto* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        if (!(ifa->ifa_flags & IFF_RUNNING) || (ifa->ifa_flags & IFF_LOOPBACK)) continue;
        char host[NI_MAXHOST];
        auto* sa = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
        if (inet_ntop(AF_INET, &sa->sin_addr, host, sizeof(host))) {
            addrs.emplace_back(host);
        }
    }
    freeifaddrs(ifaddr);
#endif
    return addrs;
}

} // namespace

struct SettingsSnapshot {
    int decimate{config::DEFAULT_DECIMATE};
    bool clahe{true};
    double gamma{1.25};
    double emaPos{config::EMA_ALPHA_POS};
    double emaPose{config::EMA_ALPHA_POSE};
    bool publishNT{true};
    bool highSpeed{config::DEFAULT_HIGH_SPEED_MODE};
    bool fastPreview{true};
    bool roiOverlay{true};
    bool grayPreview{false};
    bool diagnostics{true};
    bool autoExposure{true};
    int exposureSlider{static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE * 0.55)};
    int gainSlider{static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE * 0.3)};
    int brightnessSlider{static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE * 0.5)};
    bool udp{true};
    std::string ntServer{config::NT_SERVER};
};

class VisionApp {
public:
    VisionApp();
    ~VisionApp();

    void start();
    void stop();

    SettingsSnapshot getSettings() const;
    void applySettings(const SettingsSnapshot& updated);
    void resetRoi();
    void restartCameraForMode();

    struct CaptureProfile { int width; int height; int fps; };
    CaptureProfile profileForMode(bool highSpeed) const;

    ProcessingStats latestStats();
    cv::Mat latestFrame(bool withOverlay = true);

    bool isRunning() const { return running_; }

private:
    void startCamera();
    void stopCamera();
    void startProcessing();
    void stopProcessing();
    void captureLoop();
    void processingLoop();
    void syncProcessorFromSettings();
    cv::Mat composeDisplayFrame();

    std::unique_ptr<FrameProcessor> processor_;
    std::shared_ptr<NetworkPublisher> publisher_;

    LatestFrameExchange frameExchange_;

    std::unique_ptr<cv::VideoCapture> cap_;
    std::thread captureThread_;
    std::thread processingThread_;

    std::mutex visMutex_;
    cv::Mat latestProcessed_;
    ProcessingStats latestStats_;
    bool statsValid_{false};

    std::atomic<bool> running_{false};
    std::atomic<bool> captureRunning_{false};
    std::atomic<bool> processingRunning_{false};
    // UI-configurable settings
    std::atomic<int> decimate_{config::DEFAULT_DECIMATE};
    std::atomic<int> clahe_{1};
    std::atomic<int> gamma_{125};
    std::atomic<int> emaPos_{static_cast<int>(config::EMA_ALPHA_POS * 100.0)};
    std::atomic<int> emaPose_{static_cast<int>(config::EMA_ALPHA_POSE * 100.0)};
    std::atomic<int> publishNT_{1};
    std::atomic<int> highSpeed_{config::DEFAULT_HIGH_SPEED_MODE ? 1 : 0};
    std::atomic<int> roiOverlay_{1};
    std::atomic<int> grayPreview_{0};
    std::atomic<int> diagnostics_{1};
    std::atomic<int> udp_{1};

    std::atomic<int> previewFast_{1};

    std::atomic<int> autoExposure_{1};
    std::atomic<int> exposure_{static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE * 0.55)};
    std::atomic<int> gain_{static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE * 0.3)};
    std::atomic<int> brightness_{static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE * 0.5)};

    std::string ntServer_ = config::NT_SERVER;
};

VisionApp::VisionApp() {
    processor_ = std::make_unique<FrameProcessor>();
    HighSpeedConfig hsCfg;
    hsCfg.forcedSize = {config::HIGH_SPEED_WIDTH, config::HIGH_SPEED_HEIGHT};
    processor_->configureHighSpeed(hsCfg);
    processor_->setHighSpeedMode(config::DEFAULT_HIGH_SPEED_MODE);

    publisher_ = std::make_shared<NetworkPublisher>(config::NT_SERVER,
                                                    config::UDP_TARGET_IP,
                                                    config::UDP_TARGET_PORT);
    publisher_->enableNetworkTables(true);

    std::cout << "[VisionApp] Initialized with high-speed mode="
              << (config::DEFAULT_HIGH_SPEED_MODE ? "ON" : "OFF") << std::endl;
}

VisionApp::~VisionApp() {
    stop();
}

VisionApp::CaptureProfile VisionApp::profileForMode(bool highSpeed) const {
    if (highSpeed) {
        return {config::FAST_CAPTURE_WIDTH, config::FAST_CAPTURE_HEIGHT, config::FAST_CAPTURE_FPS};
    }
    return {config::FULL_CAPTURE_WIDTH, config::FULL_CAPTURE_HEIGHT, config::FULL_CAPTURE_FPS};
}

void VisionApp::start() {
    if (running_) return;
    running_ = true;

    publisher_->start();
    startCamera();
    startProcessing();
}

void VisionApp::stop() {
    running_ = false;
    stopProcessing();
    stopCamera();
    if (publisher_) {
        publisher_->stop();
    }
}

void VisionApp::restartCameraForMode() {
    stopCamera();
    frameExchange_.reset();
    startCamera();
}

SettingsSnapshot VisionApp::getSettings() const {
    SettingsSnapshot snap;
    snap.decimate = decimate_.load();
    snap.clahe = clahe_.load() > 0;
    snap.gamma = gamma_.load() / 100.0;
    snap.emaPos = emaPos_.load() / 100.0;
    snap.emaPose = emaPose_.load() / 100.0;
    snap.publishNT = publishNT_.load() > 0;
    snap.highSpeed = highSpeed_.load() > 0;
    snap.fastPreview = previewFast_.load() > 0;
    snap.roiOverlay = roiOverlay_.load() > 0;
    snap.grayPreview = grayPreview_.load() > 0;
    snap.diagnostics = diagnostics_.load() > 0;
    snap.autoExposure = autoExposure_.load() > 0;
    snap.exposureSlider = exposure_.load();
    snap.gainSlider = gain_.load();
    snap.brightnessSlider = brightness_.load();
    snap.udp = udp_.load() > 0;
    snap.ntServer = ntServer_;
    return snap;
}

void VisionApp::applySettings(const SettingsSnapshot& updated) {
    decimate_.store(std::max(1, updated.decimate));
    clahe_.store(updated.clahe ? 1 : 0);
    gamma_.store(static_cast<int>(updated.gamma * 100));
    emaPos_.store(static_cast<int>(std::clamp(updated.emaPos, 0.01, 0.99) * 100));
    emaPose_.store(static_cast<int>(std::clamp(updated.emaPose, 0.01, 0.99) * 100));
    publishNT_.store(updated.publishNT ? 1 : 0);
    const bool prevHighSpeed = highSpeed_.load() > 0;
    highSpeed_.store(updated.highSpeed ? 1 : 0);
    roiOverlay_.store(updated.roiOverlay ? 1 : 0);
    grayPreview_.store(updated.grayPreview ? 1 : 0);
    diagnostics_.store(updated.diagnostics ? 1 : 0);
    udp_.store(updated.udp ? 1 : 0);
    previewFast_.store(updated.fastPreview ? 1 : 0);
    ntServer_ = updated.ntServer.empty() ? config::NT_SERVER : updated.ntServer;

    autoExposure_.store(updated.autoExposure ? 1 : 0);
    exposure_.store(std::clamp(updated.exposureSlider, 0, static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE)));
    gain_.store(std::clamp(updated.gainSlider, 0, static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE)));
    brightness_.store(std::clamp(updated.brightnessSlider, 0, static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE)));

    publisher_->enableNetworkTables(updated.publishNT);
    publisher_->enableUDP(updated.udp);
    publisher_->setNtServer(ntServer_);
    if (running_ && prevHighSpeed != updated.highSpeed) {
        restartCameraForMode();
    }
}

void VisionApp::resetRoi() {
    processor_->setHighSpeedMode(false);
    processor_->setHighSpeedMode(highSpeed_.load() > 0);
}

ProcessingStats VisionApp::latestStats() {
    std::lock_guard<std::mutex> lock(visMutex_);
    return latestStats_;
}

cv::Mat VisionApp::latestFrame(bool withOverlay) {
    std::lock_guard<std::mutex> lock(visMutex_);
    if (latestProcessed_.empty()) return {};
    cv::Mat vis = latestProcessed_.clone();
    if (!withOverlay) return vis;

    ProcessingStats stats = latestStats_;
    if (!statsValid_) return vis;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1)
        << stats.effectiveFps << "fps | Proc " << stats.avgProcessTimeMs << "ms"
        << " | Tags " << stats.tagCount
        << " | Blur " << std::setprecision(0) << stats.blurVariance;
    if (stats.roiActive) {
        oss << " | ROI " << std::setprecision(1) << stats.roiCoverage * 100.0 << "%";
    }
    oss << " | Mode " << (stats.highSpeedMode ? "HS" : "Full");

    cv::rectangle(vis, {8, 8}, {vis.cols - 8, 40}, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(vis, oss.str(), {16, 34}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    return vis;
}

void VisionApp::startCamera() {
    if (captureRunning_) return;

    const bool highSpeedMode = highSpeed_.load() > 0;

    auto profile = profileForMode(highSpeedMode);

    camtuner::Settings settings;
    settings.index = config::CAM_IDX;
    settings.width = profile.width;
    settings.height = profile.height;
    settings.fps = profile.fps;
    settings.warmupFrames = config::CAM_WARMUP_FRAMES;

#ifdef _WIN32
    settings.backendOrder = {cv::CAP_DSHOW, cv::CAP_MSMF};
#else
    settings.backendOrder = {0};
#endif

    auto opened = camtuner::openAndTune(settings);
    if (!opened.ok || !opened.cap.isOpened()) {
        throw std::runtime_error("Unable to open camera index " + std::to_string(config::CAM_IDX));
    }

    if (config::CAM_FORCE_MJPEG) {
        opened.cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    }
    opened.cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    cap_ = std::make_unique<cv::VideoCapture>(std::move(opened.cap));
    captureRunning_ = true;
    frameExchange_.reset();
    captureThread_ = std::thread(&VisionApp::captureLoop, this);

    std::cout << "[Camera] Started (" << settings.width << "x" << settings.height
              << " @" << settings.fps << "fps) mode=" << (highSpeedMode ? "FAST" : "FULL") << std::endl;
    for (auto& line : opened.log) {
        std::cout << "  prop " << line << std::endl;
    }
}

void VisionApp::startProcessing() {
    if (processingRunning_) return;
    processingRunning_ = true;
    processingThread_ = std::thread(&VisionApp::processingLoop, this);
}

void VisionApp::stopCamera() {
    if (!captureRunning_) return;
    captureRunning_ = false;
    if (captureThread_.joinable()) {
        captureThread_.join();
    }
    if (cap_) {
        cap_->release();
        cap_.reset();
    }
}

void VisionApp::stopProcessing() {
    if (!processingRunning_) return;
    processingRunning_ = false;
    frameExchange_.shutdown();
    if (processingThread_.joinable()) {
        processingThread_.join();
    }
}

void VisionApp::captureLoop() {
    CameraPropertyRange exposureRange{config::CAMERA_EXPOSURE_MIN, config::CAMERA_EXPOSURE_MAX};
    CameraPropertyRange gainRange{config::CAMERA_GAIN_MIN, config::CAMERA_GAIN_MAX};
    CameraPropertyRange brightnessRange{config::CAMERA_BRIGHTNESS_MIN, config::CAMERA_BRIGHTNESS_MAX};

    int exposureCache = exposure_.load();
    int gainCache = gain_.load();
    int brightnessCache = brightness_.load();
    int autoExpCache = autoExposure_.load();
    auto lastApply = std::chrono::steady_clock::now();

    while (captureRunning_ && cap_ && cap_->isOpened()) {
        cv::Mat frame;
        if (!cap_->read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        frameExchange_.publish(frame);

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double, std::milli>(now - lastApply).count() < config::CAMERA_CONTROL_POLL_MS) {
            continue;
        }
        lastApply = now;

        const int autoExposure = autoExposure_.load();
        if (autoExposure != autoExpCache) {
            cap_->set(cv::CAP_PROP_AUTO_EXPOSURE, autoExposure ? 0.75 : 0.25);
            autoExpCache = autoExposure;
        }

        if (!autoExposure) {
            const int exposureSlider = exposure_.load();
            if (exposureSlider != exposureCache) {
                const double exposureValue = sliderToPropertyValue(exposureSlider, exposureRange);
                cap_->set(cv::CAP_PROP_EXPOSURE, exposureValue);
                exposureCache = exposureSlider;
            }
        }

        const int gainSlider = gain_.load();
        if (gainSlider != gainCache) {
            cap_->set(cv::CAP_PROP_GAIN, sliderToPropertyValue(gainSlider, gainRange));
            gainCache = gainSlider;
        }

        const int brightnessSlider = brightness_.load();
        if (brightnessSlider != brightnessCache) {
            cap_->set(cv::CAP_PROP_BRIGHTNESS, sliderToPropertyValue(brightnessSlider, brightnessRange));
            brightnessCache = brightnessSlider;
        }
    }
}

void VisionApp::processingLoop() {
    cv::Mat frame;
    auto lastLog = std::chrono::steady_clock::now();
    while (processingRunning_) {
        if (!frameExchange_.acquire(frame, std::chrono::milliseconds(10))) {
            continue;
        }
        if (frame.empty()) {
            continue;
        }

        syncProcessorFromSettings();

        ProcessingStats stats;
        cv::Mat vis = processor_->processFrame(frame, stats);
        if (vis.empty()) {
            vis = frame.clone();
        }

        if (grayPreview_.load() > 0 && vis.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(vis, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        }

        if (roiOverlay_.load() > 0 && stats.roiActive) {
            cv::rectangle(vis, stats.roiRect, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
        }

        {
            std::lock_guard<std::mutex> lock(visMutex_);
            latestProcessed_ = vis;
            latestStats_ = stats;
            statsValid_ = true;
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - lastLog).count() >= config::FPS_LOG_INTERVAL_S) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1)
                << "[Perf] " << stats.effectiveFps << " fps | Proc " << stats.avgProcessTimeMs << " ms"
                << " | Tags " << stats.tagCount
                << " | Blur " << std::setprecision(0) << stats.blurVariance;
            if (stats.roiActive) {
                oss << " | ROI " << std::setprecision(2) << stats.roiCoverage * 100.0 << "%";
            }
            std::cout << oss.str() << std::endl;
            lastLog = now;
        }
    }
}

void VisionApp::syncProcessorFromSettings() {
    processor_->setDecimate(std::max(1, decimate_.load()));
    processor_->enableCLAHE(clahe_.load() > 0);
    processor_->setGamma(std::max(0.1, gamma_.load() / 100.0));
    const double posAlpha = std::max(0.01, emaPos_.load() / 100.0);
    const double poseAlphaBase = std::max(0.01, emaPose_.load() / 100.0);
    const double poseAlpha = poseAlphaBase + (processor_->isHighSpeedMode() ? config::HIGH_SPEED_ACCURACY_BONUS : 0.0);
    processor_->setEMAAlpha(posAlpha, std::min(0.95, poseAlpha));

    const bool publish = publishNT_.load() > 0;
    publisher_->enableNetworkTables(publish);
    publisher_->enableUDP(udp_.load() > 0);

    const bool hs = highSpeed_.load() > 0;
    processor_->setHighSpeedMode(hs);
}

cv::Mat VisionApp::composeDisplayFrame() {
    cv::Mat vis;
    ProcessingStats stats;
    bool statsReady = false;
    {
        std::lock_guard<std::mutex> lock(visMutex_);
        if (!latestProcessed_.empty()) {
            vis = latestProcessed_.clone();
            stats = latestStats_;
            statsReady = statsValid_;
        }
    }

    if (vis.empty()) {
        vis = cv::Mat(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        cv::putText(vis, "Waiting for frames...", {40, 240}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 200, 255), 2);
        return vis;
    }

    if (statsReady && diagnostics_.load() > 0) {
        std::ostringstream diag;
        diag << "Decimate=" << decimate_.load()
             << " Gamma=" << gamma_.load() / 100.0
             << " EMApos=" << emaPos_.load() / 100.0
             << " EMApose=" << emaPose_.load() / 100.0;
        cv::putText(vis, diag.str(), {16, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        if (stats.hasPose) {
            std::ostringstream pose;
            pose << std::fixed << std::setprecision(3)
                 << "X=" << stats.poseTvec[0] << "m  Y=" << stats.poseTvec[1]
                 << "m  Z=" << stats.poseTvec[2] << "m";
            cv::putText(vis, pose.str(), {16, 84}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(140, 225, 255), 1, cv::LINE_AA);

            std::ostringstream rpy;
            rpy << std::fixed << std::setprecision(1)
                << "Roll=" << stats.poseEuler[0] << "°  Pitch=" << stats.poseEuler[1]
                << "°  Yaw=" << stats.poseEuler[2] << "°";
            cv::putText(vis, rpy.str(), {16, 106}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(140, 225, 255), 1, cv::LINE_AA);
        }
    }

    return vis;
}

class WebDashboard {
public:
    WebDashboard(VisionApp& app, const std::string& bind, int port)
        : app_(app), bind_(bind), port_(port) {}

    void start();
    void stop();

private:
    static bool paramOn(const httplib::Request& req, const std::string& key, bool defaultValue = false) {
        if (!req.has_param(key)) return defaultValue;
        const auto& v = req.get_param_value(key);
        return v == "1" || v == "true" || v == "on" || v == "yes";
    }

    static int paramInt(const httplib::Request& req, const std::string& key, int fallback) {
        if (!req.has_param(key)) return fallback;
        try {
            return std::stoi(req.get_param_value(key));
        } catch (...) {
            return fallback;
        }
    }

    static std::string dashboardHtml();

    void handleState(const httplib::Request& req, httplib::Response& res);
    void handleSettings(const httplib::Request& req, httplib::Response& res);

    VisionApp& app_;
    std::string bind_;
    int port_;
    httplib::Server server_;
    std::thread serverThread_;
    std::atomic<bool> running_{false};
};

void WebDashboard::start() {
    if (running_) return;
    running_ = true;

    server_.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(dashboardHtml(), "text/html");
    });

    server_.Get("/api/state", [&](const httplib::Request& req, httplib::Response& res) {
        handleState(req, res);
    });

    server_.Post("/api/settings", [&](const httplib::Request& req, httplib::Response& res) {
        handleSettings(req, res);
    });

    server_.Get("/api/reset_roi", [&](const httplib::Request&, httplib::Response& res) {
        app_.resetRoi();
        res.set_content("{\"status\":\"roi-reset\"}", "application/json");
    });

    server_.Get("/stream.mjpg", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache, no-store, must-revalidate");
        res.set_header("Pragma", "no-cache");
        res.set_header("Connection", "close");
        const bool fastView = req.has_param("view") && req.get_param_value("view") == "fast";
        res.set_chunked_content_provider("multipart/x-mixed-replace; boundary=frame",
            [this, fastView](size_t, httplib::DataSink& sink) {
                const auto frameDelay = std::chrono::milliseconds(static_cast<int>(1000.0 / std::max(1, config::MJPEG_STREAM_FPS)));
                while (running_) {
                    cv::Mat frame = app_.latestFrame(true);
                    if (frame.empty()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        continue;
                    }
                    cv::Mat streamFrame = frame;
                    cv::Mat resized;
                    if (fastView && config::STREAM_FAST_SCALE < 0.99) {
                        cv::resize(frame, resized, {}, config::STREAM_FAST_SCALE, config::STREAM_FAST_SCALE, cv::INTER_AREA);
                        streamFrame = resized;
                    }
                    std::vector<uchar> buf;
                    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
                    cv::imencode(".jpg", streamFrame, buf, params);
                    std::string header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + std::to_string(buf.size()) + "\r\n\r\n";
                    sink.os.write(header.c_str(), static_cast<std::streamsize>(header.size()));
                    sink.os.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
                    sink.os.write("\r\n", 2);
                    sink.os.flush();
                    std::this_thread::sleep_for(frameDelay);
                }
                sink.done();
                return true;
            });
    });

    serverThread_ = std::thread([this]() {
        std::cout << "[Web] Dashboard listening on http://" << bind_ << ":" << port_ << std::endl;
        server_.listen(bind_.c_str(), port_);
    });
}

void WebDashboard::stop() {
    if (!running_) return;
    running_ = false;
    server_.stop();
    if (serverThread_.joinable()) {
        serverThread_.join();
    }
}

void WebDashboard::handleState(const httplib::Request&, httplib::Response& res) {
    auto stats = app_.latestStats();
    auto settings = app_.getSettings();

    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{";
    json << "\"fps\":" << stats.effectiveFps << ",";
    json << "\"proc_ms\":" << stats.avgProcessTimeMs << ",";
    json << "\"tags\":" << stats.tagCount << ",";
    json << "\"blur\":" << stats.blurVariance << ",";
    json << "\"roiActive\":" << (stats.roiActive ? "true" : "false") << ",";
    json << "\"roiCoverage\":" << stats.roiCoverage << ",";
    json << "\"mode\":\"" << (stats.highSpeedMode ? "high-speed" : "full" ) << "\",";
    json << "\"pose\":{";
    json << "\"has\":" << (stats.hasPose ? "true" : "false");
    if (stats.hasPose) {
        json << ",\"x\":" << stats.poseTvec[0]
             << ",\"y\":" << stats.poseTvec[1]
             << ",\"z\":" << stats.poseTvec[2]
             << ",\"roll\":" << stats.poseEuler[0]
             << ",\"pitch\":" << stats.poseEuler[1]
             << ",\"yaw\":" << stats.poseEuler[2];
    }
    json << "},";
    json << "\"settings\":{"
         << "\"decimate\":" << settings.decimate << ","
         << "\"clahe\":" << (settings.clahe ? "true" : "false") << ","
         << "\"gamma\":" << settings.gamma << ","
         << "\"emaPos\":" << settings.emaPos << ","
         << "\"emaPose\":" << settings.emaPose << ","
         << "\"publishNT\":" << (settings.publishNT ? "true" : "false") << ","
         << "\"highSpeed\":" << (settings.highSpeed ? "true" : "false") << ","
         << "\"fastPreview\":" << (settings.fastPreview ? "true" : "false") << ","
         << "\"roiOverlay\":" << (settings.roiOverlay ? "true" : "false") << ","
         << "\"grayPreview\":" << (settings.grayPreview ? "true" : "false") << ","
         << "\"diagnostics\":" << (settings.diagnostics ? "true" : "false") << ","
         << "\"udp\":" << (settings.udp ? "true" : "false") << ","
         << "\"autoExposure\":" << (settings.autoExposure ? "true" : "false") << ","
         << "\"exposure\":" << settings.exposureSlider << ","
         << "\"gain\":" << settings.gainSlider << ","
         << "\"brightness\":" << settings.brightnessSlider << ","
         << "\"ntServer\":\"" << settings.ntServer << "\"";
    json << "},";
    json << "\"ips\":[";
    auto ips = localIpAddresses();
    for (size_t i = 0; i < ips.size(); ++i) {
        if (i > 0) json << ",";
        json << "\"" << ips[i] << "\"";
    }
    json << "]}";

    res.set_content(json.str(), "application/json");
}

void WebDashboard::handleSettings(const httplib::Request& req, httplib::Response& res) {
    auto snap = app_.getSettings();
    snap.highSpeed = paramOn(req, "highSpeed", snap.highSpeed);
    snap.fastPreview = paramOn(req, "fastPreview", snap.fastPreview);
    snap.publishNT = paramOn(req, "publishNT", snap.publishNT);
    snap.roiOverlay = paramOn(req, "roiOverlay", snap.roiOverlay);
    snap.grayPreview = paramOn(req, "grayPreview", snap.grayPreview);
    snap.clahe = paramOn(req, "clahe", snap.clahe);
    snap.diagnostics = paramOn(req, "diagnostics", snap.diagnostics);
    snap.udp = paramOn(req, "udp", snap.udp);
    snap.autoExposure = paramOn(req, "autoExposure", snap.autoExposure);

    snap.decimate = paramInt(req, "decimate", snap.decimate);
    snap.gamma = paramInt(req, "gamma", static_cast<int>(snap.gamma * 100)) / 100.0;
    snap.emaPos = paramInt(req, "emaPos", static_cast<int>(snap.emaPos * 100)) / 100.0;
    snap.emaPose = paramInt(req, "emaPose", static_cast<int>(snap.emaPose * 100)) / 100.0;
    snap.exposureSlider = paramInt(req, "exposure", snap.exposureSlider);
    snap.gainSlider = paramInt(req, "gain", snap.gainSlider);
    snap.brightnessSlider = paramInt(req, "brightness", snap.brightnessSlider);
    if (req.has_param("ntServer")) {
        snap.ntServer = req.get_param_value("ntServer");
    }

    app_.applySettings(snap);
    res.set_content("{\"status\":\"ok\"}", "application/json");
}

std::string WebDashboard::dashboardHtml() {
    std::ostringstream html;
    html << "<!DOCTYPE html><html><head><meta charset='utf-8'><title>" << config::WEB_DASHBOARD_TITLE << "</title>";
    html << "<style>body{font-family:Inter,Helvetica,Arial,sans-serif;margin:0;background:#0d1117;color:#e6edf3;}";
    html << ".top{padding:18px 24px;background:#161b22;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #30363d;}";
    html << ".card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:12px;box-shadow:0 10px 30px rgba(0,0,0,0.3);}";
    html << ".grid{display:grid;grid-template-columns:2fr 1fr;gap:12px;}";
    html << ".pill{padding:4px 10px;border-radius:12px;background:#238636;color:white;font-weight:600;font-size:12px;}";
    html << ".controls label{display:block;margin-bottom:10px;} .controls input, .controls select{width:100%;padding:8px;border-radius:8px;border:1px solid #30363d;background:#0d1117;color:#e6edf3;}";
    html << ".toggle-row{display:flex;gap:10px;flex-wrap:wrap;} .toggle{padding:8px 12px;border-radius:8px;border:1px solid #30363d;cursor:pointer;background:#0d1117;}";
    html << ".toggle.active{background:#1f6feb;border-color:#388bfd;color:white;}";
    html << "img.stream{width:100%;border-radius:10px;border:1px solid #30363d;background:#000;}";
    html << "</style></head><body>";
    html << "<div class='top'><div><div style='font-size:14px;color:#8b949e'>AprilTag Vision</div><div style='font-size:22px;font-weight:700;'>" << config::WEB_DASHBOARD_TITLE << "</div></div>";
    html << "<div id='ipList' style='font-size:14px;color:#8b949e'></div></div>";
    html << "<div class='grid'><div class='card'><div style='display:flex;align-items:center;justify-content:space-between;'>";
    html << "<div><div style='font-size:13px;color:#8b949e'>Live Stream</div><div style='font-size:18px;font-weight:700;'>Limelight-style view</div></div>";
    html << "<div class='pill' id='modePill'>High-Speed</div></div>";
    html << "<div style='margin-top:10px;'><img class='stream' src='/stream.mjpg?view=fast' /></div>";
    html << "<div id='stats' style='margin-top:12px;font-family:SFMono-Regular,Consolas,monospace;color:#e6edf3;'></div>";
    html << "<div id='pose' style='margin-top:6px;font-family:SFMono-Regular,Consolas,monospace;color:#8b949e;'></div>";
    html << "</div>";
    html << "<div><div class='card controls'><div style='font-size:13px;color:#8b949e'>Pipeline Controls</div>";
    html << "<div class='toggle-row'>"
         << "<div class='toggle' id='toggleHigh'>High-Speed</div>"
         << "<div class='toggle' id='toggleFull'>Full-Quality</div>"
         << "<div class='toggle' id='previewFast'>Preview: Fast</div>"
         << "<div class='toggle' id='previewFull'>Preview: Full</div>"
         << "<div class='toggle' id='toggleNT'>NetworkTables</div>"
         << "<div class='toggle' id='toggleUDP'>UDP</div>"
         << "<div class='toggle' id='toggleROI'>ROI Overlay</div>"
         << "<div class='toggle' id='toggleGray'>Gray Preview</div>"
         << "<div class='toggle' id='toggleCLAHE'>CLAHE</div>"
         << "<div class='toggle' id='toggleAuto'>Auto Exposure</div>"
         << "<div class='toggle' id='toggleDiag'>Diagnostics</div>"
         << "</div>";
    html << "<label>Decimate<input type='number' id='decimate' min='1' max='5' /></label>";
    html << "<label>Gamma<input type='number' id='gamma' step='0.05' min='0.5' max='3.0' /></label>";
    html << "<label>EMA Position<input type='number' id='emaPos' step='0.01' min='0.01' max='0.99' /></label>";
    html << "<label>EMA Pose<input type='number' id='emaPose' step='0.01' min='0.01' max='0.99' /></label>";
    html << "<label>Exposure<input type='range' id='exposure' min='0' max='" << static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE) << "' /></label>";
    html << "<label>Gain<input type='range' id='gain' min='0' max='" << static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE) << "' /></label>";
    html << "<label>Brightness<input type='range' id='brightness' min='0' max='" << static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE) << "' /></label>";
    html << "<label>NT Server<input type='text' id='ntServer' placeholder='10.xx.yy.2' /></label>";
    html << "<div style='display:flex;gap:10px;'><button id='saveBtn' style='flex:1;padding:10px;border-radius:8px;border:none;background:#1f6feb;color:white;font-weight:700;cursor:pointer;'>Apply</button>";
    html << "<button id='resetBtn' style='padding:10px;border-radius:8px;border:1px solid #30363d;background:#0d1117;color:#e6edf3;cursor:pointer;'>Reset ROI</button></div>";
    html << "</div></div></div>";

    html << "<script>\nconst DASH_PORT=" << config::WEB_PORT << ";\nconst toggles=[['toggleHigh','highSpeed'],['toggleFull','fullQuality'],['toggleNT','publishNT'],['toggleUDP','udp'],['toggleROI','roiOverlay'],['toggleGray','grayPreview'],['toggleCLAHE','clahe'],['toggleAuto','autoExposure'],['toggleDiag','diagnostics'],['previewFast','fastPreview'],['previewFull','fullPreview']];\nlet state={};\nfunction setToggle(id,on){const el=document.getElementById(id);if(!el)return;el.classList.toggle('active',on);}\nfunction bumpStream(){const img=document.querySelector('img.stream');if(!img)return;const mode=state.fastPreview?'fast':'full';img.src=`/stream.mjpg?view=${mode}&t=${Date.now()}`;setToggle('previewFast',mode==='fast');setToggle('previewFull',mode==='full');}\nfunction refresh(){fetch('/api/state').then(r=>r.json()).then(js=>{state=js.settings;document.getElementById('decimate').value=js.settings.decimate;document.getElementById('gamma').value=js.settings.gamma;document.getElementById('emaPos').value=js.settings.emaPos;document.getElementById('emaPose').value=js.settings.emaPose;document.getElementById('exposure').value=js.settings.exposure;document.getElementById('gain').value=js.settings.gain;document.getElementById('brightness').value=js.settings.brightness;document.getElementById('ntServer').value=js.settings.ntServer;setToggle('toggleHigh',js.settings.highSpeed);setToggle('toggleFull',!js.settings.highSpeed);setToggle('toggleNT',js.settings.publishNT);setToggle('toggleUDP',js.settings.udp);setToggle('toggleROI',js.settings.roiOverlay);setToggle('toggleGray',js.settings.grayPreview);setToggle('toggleCLAHE',js.settings.clahe);setToggle('toggleAuto',js.settings.autoExposure);setToggle('toggleDiag',js.settings.diagnostics);state.fastPreview=js.settings.fastPreview;bumpStream();document.getElementById('modePill').innerText=js.mode==='high-speed'?'High-Speed':'Full Quality';document.getElementById('stats').innerText=`${js.fps.toFixed(1)} fps | ${js.proc_ms.toFixed(2)} ms | tags ${js.tags} | blur ${js.blur.toFixed(1)} | ROI ${Math.round(js.roiCoverage*100)}%`;document.getElementById('pose').innerText=js.pose.has?`XYZ ${js.pose.x.toFixed(3)}, ${js.pose.y.toFixed(3)}, ${js.pose.z.toFixed(3)} | RPY ${js.pose.roll.toFixed(1)}, ${js.pose.pitch.toFixed(1)}, ${js.pose.yaw.toFixed(1)}`:'Pose pending';document.getElementById('ipList').innerText='IPs: '+js.ips.map(ip=>`http://${ip}:${DASH_PORT}`).join('  ');});}\nrefresh();setInterval(refresh,650);\nfunction collect(){const p=new URLSearchParams();p.set('decimate',document.getElementById('decimate').value);p.set('gamma',Math.max(0.5,document.getElementById('gamma').value));p.set('emaPos',document.getElementById('emaPos').value);p.set('emaPose',document.getElementById('emaPose').value);p.set('exposure',document.getElementById('exposure').value);p.set('gain',document.getElementById('gain').value);p.set('brightness',document.getElementById('brightness').value);p.set('ntServer',document.getElementById('ntServer').value);p.set('highSpeed',state.highSpeed?'1':'0');p.set('fastPreview',state.fastPreview?'1':'0');p.set('publishNT',state.publishNT?'1':'0');p.set('roiOverlay',state.roiOverlay?'1':'0');p.set('grayPreview',state.grayPreview?'1':'0');p.set('clahe',state.clahe?'1':'0');p.set('udp',state.udp?'1':'0');p.set('diagnostics',state.diagnostics?'1':'0');p.set('autoExposure',state.autoExposure?'1':'0');return p;}\nfunction push(){fetch('/api/settings',{method:'POST',body:collect()}).then(()=>{bumpStream();refresh();});}\nfor(const [id,key] of toggles){const el=document.getElementById(id);if(!el)continue;el.addEventListener('click',()=>{if(key==='highSpeed'){state.highSpeed=true;}else if(key==='fullQuality'){state.highSpeed=false;}else if(key==='fastPreview'){state.fastPreview=true;}else if(key==='fullPreview'){state.fastPreview=false;}else{state[key]=!state[key];}push();setTimeout(refresh,200);});}\ndocument.getElementById('saveBtn').addEventListener('click',()=>{push();});document.getElementById('resetBtn').addEventListener('click',()=>fetch('/api/reset_roi').then(()=>setTimeout(refresh,200)));\n</script>";

    html << "</body></html>";
    return html.str();
}

std::atomic<bool> gQuit{false};

void handleSignal(int) {
    gQuit = true;
}

int main(int, char**) {
    std::cout << "==================================" << std::endl;
    std::cout << "AprilTag Vision System" << std::endl;
    std::cout << "Headless Web Dashboard Runtime" << std::endl;
    std::cout << "==================================" << std::endl;

    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    try {
        VisionApp app;
        app.start();

        WebDashboard web(app, config::WEB_BIND_ADDRESS, config::WEB_PORT);
        web.start();

        auto ips = localIpAddresses();
        if (ips.empty()) {
            std::cout << "[Net] Browse to http://localhost:" << config::WEB_PORT << " for the dashboard" << std::endl;
        } else {
            std::cout << "[Net] Dashboard reachable at:" << std::endl;
            for (const auto& ip : ips) {
                std::cout << "  http://" << ip << ":" << config::WEB_PORT << std::endl;
            }
        }

        while (!gQuit.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        web.stop();
        app.stop();
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
