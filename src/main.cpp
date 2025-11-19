#include "FrameProcessor.h"
#include "NetworkPublisher.h"
#include "Config.h"
#include "CameraTuner.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
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
} // namespace

class VisionApp {
public:
    VisionApp() {
        processor_ = std::make_unique<FrameProcessor>();
        HighSpeedConfig hsCfg;
        hsCfg.forcedSize = {config::HIGH_SPEED_WIDTH, config::HIGH_SPEED_HEIGHT};
        processor_->configureHighSpeed(hsCfg);
        processor_->setHighSpeedMode(config::DEFAULT_HIGH_SPEED_MODE);

        publisher_ = std::make_shared<NetworkPublisher>(
            config::NT_SERVER,
            config::UDP_TARGET_IP,
            config::UDP_TARGET_PORT);
        processor_->setNetworkPublisher(publisher_);

        std::cout << "[VisionApp] Initialized with high-speed mode="
                  << (config::DEFAULT_HIGH_SPEED_MODE ? "ON" : "OFF") << std::endl;
    }

    ~VisionApp() {
        shutdown();
    }

    void run() {
        running_ = true;
        buildUi();

        publisher_->start();
        startCamera();
        startProcessing();

        std::cout << "\nControls:" << std::endl;
        std::cout << "  Q / ESC - Quit" << std::endl;
        std::cout << "  SPACE   - Toggle high-speed mode" << std::endl;
        std::cout << "  R       - Reset ROI tracker" << std::endl;
        std::cout << "  D       - Toggle diagnostics overlay" << std::endl;
        std::cout << std::endl;

        while (running_) {
            updateProcessorFromUI();
            cv::Mat display = composeDisplayFrame();
            cv::imshow(feedWindow_, display);

            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                running_ = false;
            } else if (key == ' ') {
                const bool nextState = !processor_->isHighSpeedMode();
                trackbars_.highSpeed.store(nextState ? 1 : 0);
                processor_->setHighSpeedMode(nextState);
                cv::setTrackbarPos("High-Speed", controlsWindow_.c_str(), nextState ? 1 : 0);
            } else if (key == 'r' || key == 'R') {
                processor_->setHighSpeedMode(false);
                processor_->setHighSpeedMode(true);
            } else if (key == 'd' || key == 'D') {
                const int current = diagnostics_.load();
                diagnostics_.store(current ? 0 : 1);
            }
        }

        shutdown();
    }

private:
    struct TrackbarState {
        std::atomic<int> decimate{config::DEFAULT_DECIMATE};
        std::atomic<int> clahe{1};
        std::atomic<int> gamma{125};
        std::atomic<int> emaPos{static_cast<int>(config::EMA_ALPHA_POS * 100.0)};
        std::atomic<int> emaPose{static_cast<int>(config::EMA_ALPHA_POSE * 100.0)};
        std::atomic<int> publishNT{1};
        std::atomic<int> highSpeed{config::DEFAULT_HIGH_SPEED_MODE ? 1 : 0};
        std::atomic<int> roiDebug{1};
        std::atomic<int> grayPreview{0};
    } trackbars_;

    struct CameraControlState {
        std::atomic<int> exposure{static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE * 0.55)};
        std::atomic<int> gain{static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE * 0.3)};
        std::atomic<int> brightness{static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE * 0.5)};
        std::atomic<int> autoExposure{1};
    } cameraControls_;

    struct TrackbarParam {
        std::string name;
        std::string window;
        std::atomic<int>* target;
        int minValue;
        int maxValue;
    };
    std::vector<TrackbarParam> trackbarParams_;

    void buildUi() {
        cv::namedWindow(controlsWindow_, cv::WINDOW_AUTOSIZE);
        cv::namedWindow(feedWindow_, cv::WINDOW_NORMAL);

        trackbarParams_ = {
            {"Decimate", controlsWindow_, &trackbars_.decimate, 1, 5},
            {"CLAHE", controlsWindow_, &trackbars_.clahe, 0, 1},
            {"Gamma x100", controlsWindow_, &trackbars_.gamma, 10, 300},
            {"EMA Pos x100", controlsWindow_, &trackbars_.emaPos, 1, 100},
            {"EMA Pose x100", controlsWindow_, &trackbars_.emaPose, 1, 100},
            {"Publish NT", controlsWindow_, &trackbars_.publishNT, 0, 1},
            {"High-Speed", controlsWindow_, &trackbars_.highSpeed, 0, 1},
            {"ROI Overlay", controlsWindow_, &trackbars_.roiDebug, 0, 1},
            {"Gray Preview", controlsWindow_, &trackbars_.grayPreview, 0, 1},
            {"Auto Exposure", controlsWindow_, &cameraControls_.autoExposure, 0, 1},
            {"Exposure", controlsWindow_, &cameraControls_.exposure, 0, static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE)},
            {"Gain", controlsWindow_, &cameraControls_.gain, 0, static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE)},
            {"Brightness", controlsWindow_, &cameraControls_.brightness, 0, static_cast<int>(config::CAMERA_PROP_SLIDER_SCALE)}
        };

        auto cb = [](int value, void* userdata) {
            if (!userdata) return;
            auto* param = static_cast<TrackbarParam*>(userdata);
            const int clamped = std::clamp(value, param->minValue, param->maxValue);
            if (clamped != value) {
                cv::setTrackbarPos(param->name.c_str(), param->window.c_str(), clamped);
            }
            param->target->store(clamped, std::memory_order_relaxed);
        };

        for (auto& p : trackbarParams_) {
            cv::createTrackbar(p.name.c_str(), p.window.c_str(), nullptr, p.maxValue, cb, &p);
            cv::setTrackbarPos(p.name.c_str(), p.window.c_str(), p.target->load());
        }
    }

    void startCamera() {
        if (captureRunning_) return;

        camtuner::Settings settings;
        settings.index = config::CAM_IDX;
        settings.width = config::HIGH_SPEED_WIDTH;
        settings.height = config::HIGH_SPEED_HEIGHT;
        settings.fps = config::CAPTURE_FPS;
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
                  << " @" << settings.fps << "fps)" << std::endl;
        for (auto& line : opened.log) {
            std::cout << "  prop " << line << std::endl;
        }
    }

    void startProcessing() {
        if (processingRunning_) return;
        processingRunning_ = true;
        processingThread_ = std::thread(&VisionApp::processingLoop, this);
    }

    void stopCamera() {
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

    void stopProcessing() {
        if (!processingRunning_) return;
        processingRunning_ = false;
        frameExchange_.shutdown();
        if (processingThread_.joinable()) {
            processingThread_.join();
        }
    }

    void shutdown() {
        stopCamera();
        stopProcessing();
        if (publisher_) {
            publisher_->stop();
        }
    }

    void captureLoop() {
        CameraPropertyRange exposureRange{config::CAMERA_EXPOSURE_MIN, config::CAMERA_EXPOSURE_MAX};
        CameraPropertyRange gainRange{config::CAMERA_GAIN_MIN, config::CAMERA_GAIN_MAX};
        CameraPropertyRange brightnessRange{config::CAMERA_BRIGHTNESS_MIN, config::CAMERA_BRIGHTNESS_MAX};

        int exposureCache = cameraControls_.exposure.load();
        int gainCache = cameraControls_.gain.load();
        int brightnessCache = cameraControls_.brightness.load();
        int autoExpCache = cameraControls_.autoExposure.load();
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

            const int autoExposure = cameraControls_.autoExposure.load();
            if (autoExposure != autoExpCache) {
                cap_->set(cv::CAP_PROP_AUTO_EXPOSURE, autoExposure ? 0.75 : 0.25);
                autoExpCache = autoExposure;
            }

            if (!autoExposure) {
                const int exposureSlider = cameraControls_.exposure.load();
                if (exposureSlider != exposureCache) {
                    const double exposureValue = sliderToPropertyValue(exposureSlider, exposureRange);
                    cap_->set(cv::CAP_PROP_EXPOSURE, exposureValue);
                    exposureCache = exposureSlider;
                }
            }

            const int gainSlider = cameraControls_.gain.load();
            if (gainSlider != gainCache) {
                cap_->set(cv::CAP_PROP_GAIN, sliderToPropertyValue(gainSlider, gainRange));
                gainCache = gainSlider;
            }

            const int brightnessSlider = cameraControls_.brightness.load();
            if (brightnessSlider != brightnessCache) {
                cap_->set(cv::CAP_PROP_BRIGHTNESS, sliderToPropertyValue(brightnessSlider, brightnessRange));
                brightnessCache = brightnessSlider;
            }
        }
    }

    void processingLoop() {
        cv::Mat frame;
        auto lastLog = std::chrono::steady_clock::now();
        while (processingRunning_) {
            if (!frameExchange_.acquire(frame, std::chrono::milliseconds(10))) {
                continue;
            }
            if (frame.empty()) {
                continue;
            }

            ProcessingStats stats;
            cv::Mat vis = processor_->processFrame(frame, stats);
            if (vis.empty()) {
                vis = frame.clone();
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

    void updateProcessorFromUI() {
        processor_->setDecimate(std::max(1, trackbars_.decimate.load()));
        processor_->enableCLAHE(trackbarOn(trackbars_.clahe));
        processor_->setGamma(std::max(0.1, trackbars_.gamma.load() / 100.0));
        processor_->setEMAAlpha(
            std::max(0.01, trackbars_.emaPos.load() / 100.0),
            std::max(0.01, trackbars_.emaPose.load() / 100.0));
        const bool publish = trackbars_.publishNT.load() > 0;
        publisher_->enableNetworkTables(publish);
        const bool hs = trackbars_.highSpeed.load() > 0;
        processor_->setHighSpeedMode(hs);
        showRoi_.store(trackbars_.roiDebug.load() > 0 ? 1 : 0);
        showGray_.store(trackbars_.grayPreview.load() > 0 ? 1 : 0);
    }

    bool trackbarOn(const std::atomic<int>& v) const {
        return v.load(std::memory_order_relaxed) > 0;
    }

    cv::Mat composeDisplayFrame() {
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

        if (showGray_.load() > 0 && vis.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(vis, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        }

        if (statsReady && showRoi_.load() > 0 && stats.roiActive) {
            cv::rectangle(vis, stats.roiRect, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
        }

        if (statsReady) {
            overlayStats(vis, stats);
        }

        return vis;
    }

    void overlayStats(cv::Mat& frame, const ProcessingStats& stats) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1)
            << stats.effectiveFps << "fps | Proc " << stats.avgProcessTimeMs << "ms"
            << " | Tags " << stats.tagCount
            << " | Blur " << std::setprecision(0) << stats.blurVariance;
        if (stats.roiActive) {
            oss << " | ROI " << std::setprecision(1) << stats.roiCoverage * 100.0 << "%";
        }
        oss << " | Mode " << (stats.highSpeedMode ? "HS" : "Full");

        std::string text = oss.str();
        cv::rectangle(frame, {8, 8}, {frame.cols - 8, 40}, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, text, {16, 34}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        if (diagnostics_.load() > 0) {
            std::ostringstream diag;
            diag << "Decimate=" << trackbars_.decimate.load()
                 << " Gamma=" << trackbars_.gamma.load() / 100.0
                 << " EMApos=" << trackbars_.emaPos.load() / 100.0
                 << " EMApose=" << trackbars_.emaPose.load() / 100.0;
            cv::putText(frame, diag.str(), {16, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

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

    std::atomic<int> showGray_{0};
    std::atomic<int> showRoi_{1};
    std::atomic<int> diagnostics_{1};

    const std::string controlsWindow_ = "AprilTag Controls";
    const std::string feedWindow_ = "AprilTag Vision";
};

int main(int argc, char** argv) {
    std::cout << "==================================" << std::endl;
    std::cout << "AprilTag Vision System" << std::endl;
    std::cout << "High-Speed Dual-Thread Runtime" << std::endl;
    std::cout << "==================================" << std::endl;

    try {
        VisionApp app;
        app.run();
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
