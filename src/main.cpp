#include "FrameProcessor.h"
#include "NetworkPublisher.h"
#include "Config.h"
#include "CameraTuner.h"   // <<< NEW
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <array>
#include <utility>

// Undefine Windows macros
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

class LatestFrameExchange {
public:
    struct Lease {
        cv::Mat view;
        LatestFrameExchange* owner = nullptr;
        int index = -1;

        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept {
            *this = std::move(other);
        }

        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();
                view = std::move(other.view);
                owner = other.owner;
                index = other.index;
                other.owner = nullptr;
                other.index = -1;
            }
            return *this;
        }

        ~Lease() { release(); }

        explicit operator bool() const { return owner != nullptr; }

        void release() {
            if (owner) {
                owner->release(index);
                owner = nullptr;
                index = -1;
                view.release();
            }
        }
    };

    Lease waitForFrame(uint64_t& lastSeq, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait_for(lock, timeout, [&] { return stopped_ || (seq_ != lastSeq && latestIndex_ >= 0); });
        if (stopped_ || latestIndex_ < 0 || seq_ == lastSeq) {
            return {};
        }
        borrowedIndex_ = latestIndex_;
        Lease lease;
        lease.view = buffers_[latestIndex_];
        lease.owner = this;
        lease.index = latestIndex_;
        lastSeq = seq_;
        return lease;
    }

    void push(const cv::Mat& frame) {
        if (frame.empty()) return;
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopped_) return;
        const int target = selectWriteIndexLocked();
        if (buffers_[target].empty() || buffers_[target].size() != frame.size() ||
            buffers_[target].type() != frame.type()) {
            buffers_[target].create(frame.size(), frame.type());
        }
        frame.copyTo(buffers_[target]);
        latestIndex_ = target;
        ++seq_;
        lock.unlock();
        cond_.notify_one();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        latestIndex_ = -1;
        if (borrowedIndex_ == -1) {
            for (auto& buf : buffers_) buf.release();
        }
        seq_ = 0;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cond_.notify_all();
    }

private:
    static constexpr int BUFFER_COUNT = 3;
    std::array<cv::Mat, BUFFER_COUNT> buffers_;
    int latestIndex_ = -1;
    int borrowedIndex_ = -1;
    uint64_t seq_ = 0;
    bool stopped_ = false;
    std::mutex mutex_;
    std::condition_variable cond_;

    void release(int index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (borrowedIndex_ == index) {
            borrowedIndex_ = -1;
        }
    }

    int selectWriteIndexLocked() const {
        if (latestIndex_ < 0) return 0;
        for (int i = 0; i < BUFFER_COUNT; ++i) {
            const int candidate = (latestIndex_ + 1 + i) % BUFFER_COUNT;
            if (candidate != borrowedIndex_) {
                return candidate;
            }
        }
        return (latestIndex_ + 1) % BUFFER_COUNT;
    }
};

class VisionApp {
public:
    VisionApp()
        : running_(false)
        , cameraRunning_(false)
        , detectionRunning_(false)
        , useCamera_(false)
        , highSpeedMode_(config::DEFAULT_HIGH_SPEED_MODE)
        , currentImageIndex_(0)
    {
        processor_ = std::make_unique<FrameProcessor>();
        publisher_ = std::make_shared<NetworkPublisher>(
            config::NT_SERVER,
            config::UDP_TARGET_IP,
            config::UDP_TARGET_PORT
        );
        processor_->setNetworkPublisher(publisher_);
        processor_->setHighSpeedMode(highSpeedMode_);

        std::cout << "[VisionApp] Initialized" << std::endl;
    }

    ~VisionApp() {
        stopCamera();
        stopDetectionThread();
        if (publisher_) {
            publisher_->stop();
        }
    }

    void run() {
        running_ = true;

        // Create control window
        const std::string controlsWindow = "AprilTag Vision - Controls";
        const std::string feedWindow = "AprilTag Vision - Feed";
        cv::namedWindow(controlsWindow, cv::WINDOW_AUTOSIZE);
        cv::namedWindow(feedWindow, cv::WINDOW_NORMAL);

        struct TrackbarState {
            std::atomic<int> decimate{config::DEFAULT_DECIMATE};
            std::atomic<int> clahe{1};
            std::atomic<int> gamma{125};
            std::atomic<int> emaPos{static_cast<int>(config::EMA_ALPHA_POS * 100)};
            std::atomic<int> emaPose{static_cast<int>(config::EMA_ALPHA_POSE * 100)};
            std::atomic<int> publishNT{0};
        } trackbars;

        struct TrackbarParam {
            const char* name;
            const std::string* window;
            std::atomic<int>* target;
            int minValue;
            int maxValue;
        };

        auto trackbarCallback = [](int pos, void* userdata) {
            if (!userdata) return;
            auto* param = static_cast<TrackbarParam*>(userdata);
            if (!param->target) return;
            const int clamped = std::clamp(pos, param->minValue, param->maxValue);
            if (clamped != pos) {
                cv::setTrackbarPos(param->name, *param->window, clamped);
            }
            param->target->store(clamped, std::memory_order_relaxed);
        };

        TrackbarParam decimateParam{"Decimate", &controlsWindow, &trackbars.decimate, 1, 5};
        TrackbarParam claheParam{"CLAHE", &controlsWindow, &trackbars.clahe, 0, 1};
        TrackbarParam gammaParam{"Gamma x100", &controlsWindow, &trackbars.gamma, 0, 300};
        TrackbarParam emaPosParam{"EMA Pos x100", &controlsWindow, &trackbars.emaPos, 0, 100};
        TrackbarParam emaPoseParam{"EMA Pose x100", &controlsWindow, &trackbars.emaPose, 0, 100};
        TrackbarParam publishParam{"Publish NT", &controlsWindow, &trackbars.publishNT, 0, 1};

        cv::createTrackbar(decimateParam.name, controlsWindow, nullptr, decimateParam.maxValue, trackbarCallback, &decimateParam);
        cv::createTrackbar(claheParam.name, controlsWindow, nullptr, claheParam.maxValue, trackbarCallback, &claheParam);
        cv::createTrackbar(gammaParam.name, controlsWindow, nullptr, gammaParam.maxValue, trackbarCallback, &gammaParam);
        cv::createTrackbar(emaPosParam.name, controlsWindow, nullptr, emaPosParam.maxValue, trackbarCallback, &emaPosParam);
        cv::createTrackbar(emaPoseParam.name, controlsWindow, nullptr, emaPoseParam.maxValue, trackbarCallback, &emaPoseParam);
        cv::createTrackbar(publishParam.name, controlsWindow, nullptr, publishParam.maxValue, trackbarCallback, &publishParam);

        cv::setTrackbarPos(decimateParam.name, controlsWindow, trackbars.decimate.load(std::memory_order_relaxed));
        cv::setTrackbarPos(claheParam.name, controlsWindow, trackbars.clahe.load(std::memory_order_relaxed));
        cv::setTrackbarPos(gammaParam.name, controlsWindow, trackbars.gamma.load(std::memory_order_relaxed));
        cv::setTrackbarPos(emaPosParam.name, controlsWindow, trackbars.emaPos.load(std::memory_order_relaxed));
        cv::setTrackbarPos(emaPoseParam.name, controlsWindow, trackbars.emaPose.load(std::memory_order_relaxed));
        cv::setTrackbarPos(publishParam.name, controlsWindow, trackbars.publishNT.load(std::memory_order_relaxed));

        std::cout << "\n=== Controls ===" << std::endl;
        std::cout << "C - Toggle camera mode" << std::endl;
        std::cout << "O - Open images" << std::endl;
        std::cout << "L - Load camera intrinsics" << std::endl;
        std::cout << "T - Set tag size (meters)" << std::endl;
        std::cout << "N/P - Previous/Next image (when not in camera mode)" << std::endl;
        std::cout << "H - Toggle high-speed mode (restarts camera)" << std::endl;
        std::cout << "Q/ESC - Quit" << std::endl;
        std::cout << "================\n" << std::endl;
        std::cout << "[VisionApp] High-speed mode "
                  << (highSpeedMode_ ? "ENABLED" : "DISABLED")
                  << " by default\n";

        publisher_->start();
        startDetectionThread();
        startCamera();

        cv::Mat displayFrame;
        auto lastFrameTime = std::chrono::steady_clock::now();
        const double targetFrameTime = 1.0 / config::GUI_RATE_HZ;

        while (running_) {
            const int decimateVal = trackbars.decimate.load(std::memory_order_relaxed);
            const int claheVal = trackbars.clahe.load(std::memory_order_relaxed);
            const int gammaVal = trackbars.gamma.load(std::memory_order_relaxed);
            const int emaPosVal = trackbars.emaPos.load(std::memory_order_relaxed);
            const int emaPoseVal = trackbars.emaPose.load(std::memory_order_relaxed);
            const int publishNTVal = trackbars.publishNT.load(std::memory_order_relaxed);

            {
                std::lock_guard<std::mutex> lock(processorMutex_);
                processor_->setDecimate(std::max(1, decimateVal));
                processor_->enableCLAHE(claheVal > 0);
                processor_->setGamma(std::max(0.1, gammaVal / 100.0));
                processor_->setEMAAlpha(
                    std::max(0.01, emaPosVal / 100.0),
                    std::max(0.01, emaPoseVal / 100.0)
                );
            }
            publisher_->enableNetworkTables(publishNTVal > 0);

            cv::Mat latest;
            {
                std::lock_guard<std::mutex> lock(resultMutex_);
                if (!latestResult_.empty()) {
                    latest = latestResult_;
                }
            }

            if (!latest.empty()) {
                displayFrame = latest;
            }

            cv::Mat frameToShow = displayFrame;
            if (frameToShow.empty()) {
                frameToShow = cv::Mat(480, 640, CV_8UC3, cv::Scalar(32, 32, 32));
                const bool cameraActive = cameraRunning_.load(std::memory_order_relaxed);
                const std::string message = cameraActive
                    ? "Waiting for detector frames..."
                    : "Press C to start camera or load images";
                cv::putText(frameToShow, message, cv::Point(40, 240),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 255), 2);
            }

            cv::imshow(feedWindow, frameToShow);

            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                running_ = false;
            } else if (key == 'c' || key == 'C') {
                toggleCamera();
            } else if (key == 'o' || key == 'O') {
                openImages();
            } else if (key == 'l' || key == 'L') {
                loadIntrinsics();
            } else if (key == 't' || key == 'T') {
                setTagSize();
            } else if (key == 'n' || key == 'N') {
                nextImage();
            } else if (key == 'p' || key == 'P') {
                previousImage();
            } else if (key == 'h' || key == 'H') {
                toggleHighSpeedMode();
            }

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - lastFrameTime).count();
            if (elapsed < targetFrameTime) {
                std::this_thread::sleep_for(
                    std::chrono::duration<double>(targetFrameTime - elapsed)
                );
            }
            lastFrameTime = std::chrono::steady_clock::now();
        }

        cv::destroyAllWindows();
        stopCamera();
        stopDetectionThread();
    }

private:
    void toggleCamera() {
        if (cameraRunning_) {
            stopCamera();
        } else {
            startCamera();
        }
    }

    void startDetectionThread() {
        if (detectionRunning_) return;
        frameExchange_.clear();
        detectionRunning_ = true;
        detectionThread_ = std::thread(&VisionApp::detectionLoop, this);
    }

    void stopDetectionThread() {
        if (!detectionRunning_) return;
        detectionRunning_ = false;
        frameExchange_.stop();
        if (detectionThread_.joinable()) {
            detectionThread_.join();
        }
    }

    void startCamera() {
        if (cameraRunning_) return;

        // ---- NEW: use CameraTuner for backend + exposure sanity on Windows ----
        camtuner::Settings cs;
        cs.index  = config::CAM_IDX;
        const cv::Size dims = captureSizeForMode();
        cs.width  = dims.width;
        cs.height = dims.height;
        cs.fps    = captureFpsForMode();

#ifdef _WIN32
        // Try DSHOW first, then MSMF (exposure controls are often better on DSHOW).
        cs.backendOrder = { cv::CAP_DSHOW, cv::CAP_MSMF };
        // Start with Auto Exposure ON; if still dark, the tuner flips once.
        cs.useAutoExposure = true;
        // If you prefer manual, flip these two lines instead:
        // cs.useAutoExposure = false; cs.manualExposure = -6.0; // ~1/64 s on Windows
#else
        // Non-Windows: let OpenCV choose the platform default backend
        cs.backendOrder = { 0 };
#endif
        cs.warmupFrames = config::CAM_WARMUP_FRAMES;

        frameExchange_.clear();
        {
            std::lock_guard<std::mutex> lock(resultMutex_);
            latestResult_.release();
        }

        auto opened = camtuner::openAndTune(cs);
        if (!opened.ok || !opened.cap.isOpened()) {
            std::cerr << "[VisionApp] Failed to open/tune camera " << config::CAM_IDX << std::endl;
            for (auto& line : opened.log) std::cerr << "  prop " << line << "\n";
            return;
        }

        // Force MJPEG if requested
        if (config::CAM_FORCE_MJPEG) {
            opened.cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
        }

        if (highSpeedMode_) {
            opened.cap.set(cv::CAP_PROP_CONVERT_RGB, 0.0);
        }

        // Minimal internal buffering for fresh frames
        opened.cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        cv::Mat warm;
        int warmCount = 0;
        for (int i = 0; i < cs.warmupFrames; ++i) {
            if (!opened.cap.read(warm) || warm.empty()) continue;
            frameExchange_.push(warm);
            ++warmCount;
            if (config::CAM_WARMUP_DELAY_MS > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config::CAM_WARMUP_DELAY_MS));
            }
        }

        // Move the tuned capture into our member and spawn the loop
        cap_ = std::make_unique<cv::VideoCapture>(std::move(opened.cap));
        cameraRunning_ = true;
        useCamera_ = true;
        captureThread_ = std::thread(&VisionApp::captureLoop, this);

        std::cout << "[VisionApp] Camera started";
        if (warmCount > 0) std::cout << " | warm-up frames=" << warmCount;
        std::cout << " | mode=" << (highSpeedMode_ ? "high-speed" : "full-quality") << std::endl;

        // (Optional) print what actually stuck
        std::cerr << "[Camera] Properties after tuning:\n";
        for (auto& line : opened.log) std::cerr << "  prop " << line << "\n";
    }

    void stopCamera() {
        if (!cameraRunning_) return;

        cameraRunning_ = false;
        useCamera_ = false;

        if (captureThread_.joinable()) {
            captureThread_.join();
        }

        if (cap_) {
            cap_->release();
            cap_.reset();
        }

        frameExchange_.clear();

        std::cout << "[VisionApp] Camera stopped" << std::endl;
    }

    void captureLoop() {
        while (cameraRunning_ && cap_ && cap_->isOpened()) {
            cv::Mat frame;
            if (!cap_->read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            frameExchange_.push(frame);

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void detectionLoop() {
        uint64_t lastSeq = 0;
        auto fpsWindowStart = std::chrono::steady_clock::now();
        int fpsWindowCount = 0;
        auto lastLog = std::chrono::steady_clock::now();

        while (true) {
            if (!detectionRunning_) break;
            auto lease = frameExchange_.waitForFrame(lastSeq, std::chrono::milliseconds(30));
            if (!lease) {
                if (!detectionRunning_) break;
                continue;
            }

            ProcessingStats stats;
            cv::Mat vis;
            auto frameStart = std::chrono::steady_clock::now();
            try {
                std::lock_guard<std::mutex> lock(processorMutex_);
                vis = processor_->processFrame(lease.view, stats);
            } catch (const std::exception& e) {
                std::cerr << "[VisionApp] Detection error: " << e.what() << std::endl;
                continue;
            }
            auto frameEnd = std::chrono::steady_clock::now();
            stats.frameTimeMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

            ++fpsWindowCount;
            double windowSec = std::chrono::duration<double>(frameEnd - fpsWindowStart).count();
            if (windowSec >= 2.0) {
                stats.effectiveFps = fpsWindowCount / windowSec;
                fpsWindowStart = frameEnd;
                fpsWindowCount = 0;
            } else {
                stats.effectiveFps = fpsWindowCount / std::max(1e-3, windowSec);
            }

            cv::Mat display;
            if (vis.empty()) {
                lease.view.copyTo(display);
            } else {
                display = vis;
            }

            if (!display.empty()) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(1)
                    << (stats.highSpeedMode ? "HS" : "HQ")
                    << " | FPS " << stats.effectiveFps
                    << " | Proc " << std::setprecision(2) << stats.avgProcessTimeMs << " ms"
                    << " | Tags " << stats.tagCount;
                if (stats.usedROI) {
                    oss << " | ROI " << stats.roiRect.width << "x" << stats.roiRect.height;
                } else {
                    oss << " | ROI full";
                }
                cv::putText(display, oss.str(), cv::Point(10, 24),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }

            {
                std::lock_guard<std::mutex> lock(resultMutex_);
                latestResult_ = display;
                latestStats_ = stats;
            }

            auto now = frameEnd;
            if (now - lastLog >= std::chrono::seconds(1)) {
                std::ostringstream log;
                log << "[Perf] FPS=" << std::fixed << std::setprecision(1) << stats.effectiveFps
                    << " | Proc=" << std::setprecision(2) << stats.avgProcessTimeMs
                    << " ms | Tags=" << stats.tagCount
                    << " | ROI=" << (stats.usedROI ?
                        std::to_string(stats.roiRect.width) + "x" + std::to_string(stats.roiRect.height) : "full");
                std::cout << log.str() << std::endl;
                lastLog = now;
            }
        }
    }

    void toggleHighSpeedMode() {
        highSpeedMode_ = !highSpeedMode_;
        {
            std::lock_guard<std::mutex> lock(processorMutex_);
            processor_->setHighSpeedMode(highSpeedMode_);
        }
        std::cout << "[VisionApp] High-speed mode "
                  << (highSpeedMode_ ? "ENABLED" : "DISABLED")
                  << std::endl;
        if (cameraRunning_) {
            stopCamera();
            startCamera();
        }
    }

    cv::Size captureSizeForMode() const {
        return highSpeedMode_
            ? cv::Size(config::HIGH_SPEED_FRAME_WIDTH, config::HIGH_SPEED_FRAME_HEIGHT)
            : cv::Size(config::DEFAULT_FRAME_WIDTH, config::DEFAULT_FRAME_HEIGHT);
    }

    double captureFpsForMode() const {
        return highSpeedMode_ ? config::HIGH_SPEED_TARGET_FPS : config::CAPTURE_FPS;
    }

    void openImages() {
        stopCamera();

        std::cout << "[VisionApp] Enter image paths (comma-separated): ";
        std::string input;
        std::getline(std::cin, input);

        imageFiles_.clear();
        std::istringstream iss(input);
        std::string path;

        while (std::getline(iss, path, ',')) {
            // Trim whitespace
            path.erase(0, path.find_first_not_of(" \t"));
            path.erase(path.find_last_not_of(" \t") + 1);

            if (!path.empty()) {
                imageFiles_.push_back(path);
            }
        }

        if (!imageFiles_.empty()) {
            currentImageIndex_ = 0;
            loadCurrentImage();
            useCamera_ = false;
            std::cout << "[VisionApp] Loaded " << imageFiles_.size() << " images" << std::endl;
        }
    }

    void loadCurrentImage() {
        if (imageFiles_.empty()) return;

        cv::Mat img = cv::imread(imageFiles_[currentImageIndex_]);
        if (!img.empty()) {
            frameExchange_.clear();
            frameExchange_.push(img);
            {
                std::lock_guard<std::mutex> lock(resultMutex_);
                latestResult_ = img;
            }
        } else {
            std::cerr << "[VisionApp] Failed to load: " << imageFiles_[currentImageIndex_] << std::endl;
        }
    }

    void nextImage() {
        if (imageFiles_.empty() || useCamera_) return;

        currentImageIndex_ = (currentImageIndex_ + 1) % imageFiles_.size();
        loadCurrentImage();
    }

    void previousImage() {
        if (imageFiles_.empty() || useCamera_) return;

        currentImageIndex_ = (currentImageIndex_ == 0) ?
            imageFiles_.size() - 1 : currentImageIndex_ - 1;
        loadCurrentImage();
    }

    void loadIntrinsics() {
        std::cout << "[VisionApp] Enter intrinsics JSON path: ";
        std::string path;
        std::getline(std::cin, path);

        // Trim whitespace
        path.erase(0, path.find_first_not_of(" \t"));
        path.erase(path.find_last_not_of(" \t") + 1);

        if (path.empty()) return;

        try {
            std::ifstream file(path);
            if (!file.is_open()) {
                std::cerr << "[VisionApp] Cannot open: " << path << std::endl;
                return;
            }

            // Simple JSON parsing (for production, use a proper JSON library like nlohmann/json)
            std::string line, content;
            while (std::getline(file, line)) {
                content += line;
            }

            // Extract fx, fy, cx, cy (very basic parsing)
            double fx = 0, fy = 0, cx = 0, cy = 0;

            size_t fxPos = content.find("\"fx\"");
            if (fxPos != std::string::npos) {
                sscanf(content.c_str() + fxPos, "\"fx\": %lf", &fx);
            }

            size_t fyPos = content.find("\"fy\"");
            if (fyPos != std::string::npos) {
                sscanf(content.c_str() + fyPos, "\"fy\": %lf", &fy);
            } else {
                fy = fx;
            }

            size_t cxPos = content.find("\"cx\"");
            if (cxPos != std::string::npos) {
                sscanf(content.c_str() + cxPos, "\"cx\": %lf", &cx);
            }

            size_t cyPos = content.find("\"cy\"");
            if (cyPos != std::string::npos) {
                sscanf(content.c_str() + cyPos, "\"cy\": %lf", &cy);
            }

            if (fx > 0 && fy > 0 && cx > 0 && cy > 0) {
                cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
                cv::Mat D = cv::Mat::zeros(5, 1, CV_64F);

                {
                    std::lock_guard<std::mutex> lock(processorMutex_);
                    processor_->setCameraMatrix(K, D);
                }
                std::cout << "[VisionApp] Loaded intrinsics from " << path << std::endl;
            } else {
                std::cerr << "[VisionApp] Invalid intrinsics data" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[VisionApp] Error loading intrinsics: " << e.what() << std::endl;
        }
    }

    void setTagSize() {
        std::cout << "[VisionApp] Enter tag size in meters: ";
        double size;
        std::cin >> size;
        std::cin.ignore();

        if (size > 0.0) {
            std::lock_guard<std::mutex> lock(processorMutex_);
            processor_->setTagSize(size);
            std::cout << "[VisionApp] Tag size set to " << size << " meters" << std::endl;
        }
    }

    std::unique_ptr<FrameProcessor> processor_;
    std::shared_ptr<NetworkPublisher> publisher_;
    std::mutex processorMutex_;

    std::atomic<bool> running_;
    std::atomic<bool> cameraRunning_;
    std::atomic<bool> detectionRunning_;
    std::atomic<bool> useCamera_;

    std::unique_ptr<cv::VideoCapture> cap_;
    std::thread captureThread_;
    LatestFrameExchange frameExchange_;
    std::thread detectionThread_;

    std::mutex resultMutex_;
    cv::Mat latestResult_;
    ProcessingStats latestStats_;

    bool highSpeedMode_;
    std::vector<std::string> imageFiles_;
    size_t currentImageIndex_;
};

int main(int argc, char** argv) {
    std::cout << "==================================" << std::endl;
    std::cout << "AprilTag Vision System for FRC" << std::endl;
    std::cout << "C++ High-Performance Edition" << std::endl;
    std::cout << "==================================" << std::endl;

    try {
        VisionApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
