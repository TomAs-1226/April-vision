#include "FrameProcessor.h"
#include "NetworkPublisher.h"
#include "Config.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>

// Undefine Windows macros
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

using namespace vision_config;

// ... rest of your main.cpp code stays the same
class VisionApp {
public:
    VisionApp()
        : running_(false)
        , cameraRunning_(false)
        , useCamera_(false)
        , currentImageIndex_(0)
    {
        processor_ = std::make_unique<FrameProcessor>();
        publisher_ = std::make_shared<NetworkPublisher>(
            config::NT_SERVER,
            config::UDP_TARGET_IP,
            config::UDP_TARGET_PORT
        );
        processor_->setNetworkPublisher(publisher_);

        std::cout << "[VisionApp] Initialized" << std::endl;
    }

    ~VisionApp() {
        stopCamera();
        if (publisher_) {
            publisher_->stop();
        }
    }

    void run() {
        running_ = true;

        // Create control window
        cv::namedWindow("AprilTag Vision - Controls", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("AprilTag Vision - Feed", cv::WINDOW_NORMAL);

        // Create trackbars for controls
        int decimateVal = config::DEFAULT_DECIMATE;
        int claheVal = 1;
        int gammaVal = 125; // Gamma * 100
        int emaPosVal = static_cast<int>(config::EMA_ALPHA_POS * 100);
        int emaPoseVal = static_cast<int>(config::EMA_ALPHA_POSE * 100);
        int publishNTVal = 0;

        cv::createTrackbar("Decimate", "AprilTag Vision - Controls", &decimateVal, 5);
        cv::createTrackbar("CLAHE", "AprilTag Vision - Controls", &claheVal, 1);
        cv::createTrackbar("Gamma x100", "AprilTag Vision - Controls", &gammaVal, 300);
        cv::createTrackbar("EMA Pos x100", "AprilTag Vision - Controls", &emaPosVal, 100);
        cv::createTrackbar("EMA Pose x100", "AprilTag Vision - Controls", &emaPoseVal, 100);
        cv::createTrackbar("Publish NT", "AprilTag Vision - Controls", &publishNTVal, 1);

        std::cout << "\n=== Controls ===" << std::endl;
        std::cout << "C - Toggle camera mode" << std::endl;
        std::cout << "O - Open images" << std::endl;
        std::cout << "L - Load camera intrinsics" << std::endl;
        std::cout << "T - Set tag size (meters)" << std::endl;
        std::cout << "N/P - Previous/Next image (when not in camera mode)" << std::endl;
        std::cout << "Q/ESC - Quit" << std::endl;
        std::cout << "================\n" << std::endl;

        publisher_->start();

        cv::Mat displayFrame;
        auto lastFrameTime = std::chrono::steady_clock::now();
        const double targetFrameTime = 1.0 / config::GUI_RATE_HZ;

        while (running_) {
            // Update processor settings from trackbars
            processor_->setDecimate(std::max(1, decimateVal));
            processor_->enableCLAHE(claheVal > 0);
            processor_->setGamma(std::max(0.1, gammaVal / 100.0));
            processor_->setEMAAlpha(
                std::max(0.01, emaPosVal / 100.0),
                std::max(0.01, emaPoseVal / 100.0)
            );
            publisher_->enableNetworkTables(publishNTVal > 0);

            // Get frame to process
            cv::Mat frameToProcess;
            {
                std::lock_guard<std::mutex> lock(frameMutex_);
                if (!latestFrame_.empty()) {
                    frameToProcess = latestFrame_.clone();
                }
            }

            if (!frameToProcess.empty()) {
                ProcessingStats stats;
                cv::Mat result = processor_->processFrame(frameToProcess, stats);

                if (!result.empty()) {
                    // Add stats overlay
                    std::ostringstream oss;
                    oss << "Det Rate: " << std::fixed << std::setprecision(1) << stats.detectionRateHz << " Hz | "
                        << "Proc: " << std::setprecision(1) << stats.avgProcessTimeMs << " ms | "
                        << "Tags: " << stats.tagCount << " | "
                        << "Blur: " << std::setprecision(0) << stats.blurVariance;

                    std::string statsText = oss.str();
                    cv::putText(result, statsText, cv::Point(10, 30),
                              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

                    displayFrame = result;
                }
            }

            // Display frame
            if (!displayFrame.empty()) {
                cv::imshow("AprilTag Vision - Feed", displayFrame);
            }

            // Handle keyboard input
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) { // Q or ESC
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
            }

            // Rate limiting
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
    }

private:
    void toggleCamera() {
        if (cameraRunning_) {
            stopCamera();
        } else {
            startCamera();
        }
    }

    void startCamera() {
        if (cameraRunning_) return;

        cap_ = std::make_unique<cv::VideoCapture>(config::CAM_IDX);

        if (!cap_->isOpened()) {
            std::cerr << "[VisionApp] Failed to open camera " << config::CAM_IDX << std::endl;
            return;
        }

        cap_->set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap_->set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap_->set(cv::CAP_PROP_FPS, config::CAPTURE_FPS);
        cap_->set(cv::CAP_PROP_BUFFERSIZE, 1);

        cameraRunning_ = true;
        useCamera_ = true;
        captureThread_ = std::thread(&VisionApp::captureLoop, this);

        std::cout << "[VisionApp] Camera started" << std::endl;
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

        std::cout << "[VisionApp] Camera stopped" << std::endl;
    }

    void captureLoop() {
        while (cameraRunning_ && cap_ && cap_->isOpened()) {
            cv::Mat frame;
            if (cap_->read(frame)) {
                std::lock_guard<std::mutex> lock(frameMutex_);
                latestFrame_ = frame;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void openImages() {
        // Note: OpenCV doesn't have a built-in file dialog
        // You'd need to implement this with system-specific code or use a library
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
            std::lock_guard<std::mutex> lock(frameMutex_);
            latestFrame_ = img;
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

                processor_->setCameraMatrix(K, D);
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
            processor_->setTagSize(size);
            std::cout << "[VisionApp] Tag size set to " << size << " meters" << std::endl;
        }
    }

    std::unique_ptr<FrameProcessor> processor_;
    std::shared_ptr<NetworkPublisher> publisher_;

    std::atomic<bool> running_;
    std::atomic<bool> cameraRunning_;
    std::atomic<bool> useCamera_;

    std::unique_ptr<cv::VideoCapture> cap_;
    std::thread captureThread_;

    cv::Mat latestFrame_;
    std::mutex frameMutex_;

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