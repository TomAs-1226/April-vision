#pragma once

#include "Detector.h"
#include "PoseEstimator.h"
#include "Tracker.h"
#include "NetworkPublisher.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <map>
#include <set>
#include <deque>
#include <mutex>
#include <atomic>
#include <chrono>

struct ProcessingStats {
    double detectionRateHz;
    double avgProcessTimeMs;
    int tagCount;
    double blurVariance;
};

class FrameProcessor {
public:
    FrameProcessor();
    ~FrameProcessor();

    // Process a frame and return visualization
    cv::Mat processFrame(const cv::Mat& frame, ProcessingStats& stats);

    // Configuration
    void setTagSize(double meters) { tagSizeM_ = meters; }
    void setCameraMatrix(const cv::Mat& K, const cv::Mat& D);
    void setDecimate(int dec) { baseDecimate_ = dec; }
    void setEMAAlpha(double pos, double pose);
    void enableCLAHE(bool enable) { useCLAHE_ = enable; }
    void setGamma(double gamma);
    void setWhitelist(const std::set<int>& ids) { whitelist_ = ids; useWhitelist_ = true; }
    void setBlacklist(const std::set<int>& ids) { blacklist_ = ids; useBlacklist_ = true; }
    void clearFilters() { useWhitelist_ = false; useBlacklist_ = false; }

    // Network publishing
    void setNetworkPublisher(std::shared_ptr<NetworkPublisher> pub) { publisher_ = pub; }

    double getDetectionRate() const { return detectionRate_; }

private:
    // Image preprocessing
    cv::Mat preprocessImage(const cv::Mat& gray);
    double computeBlurVariance(const cv::Mat& gray);
    int chooseDecimate(int base, double blurVar);
    void buildGammaLUT(double gamma);

    // Detection filtering
    bool shouldProcessTag(int id);

    // Tracking helpers
    void updateTracker(int id, const std::vector<cv::Point2f>& corners);
    void predictInvisibleTags(cv::Mat& vis, int width, int height,
                             const std::set<int>& visibleIds,
                             const cv::Mat& grayPrev, const cv::Mat& grayCurr);
    void purgeOldTrackers(const std::set<int>& visibleIds);

    // Visualization
    void drawDetection(cv::Mat& vis, const Detection& det,
                      const std::vector<cv::Point2f>& corners,
                      double tx, double ty, double ta, const cv::Vec3d& tvec);
    void drawPrediction(cv::Mat& vis, int id, double cx, double cy, double s, bool isOpticalFlow);

    // Members
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<PoseEstimator> poseEstimator_;
    std::shared_ptr<NetworkPublisher> publisher_;

    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    double tagSizeM_;

    // Preprocessing
    bool useCLAHE_;
    cv::Ptr<cv::CLAHE> clahe_;
    double gamma_;
    std::vector<uint8_t> gammaLUT_;
    int baseDecimate_;

    // Filters
    bool useWhitelist_;
    bool useBlacklist_;
    std::set<int> whitelist_;
    std::set<int> blacklist_;

    // Tracking state
    std::map<int, std::unique_ptr<BoxTracker>> trackers_;
    std::map<int, std::unique_ptr<EMASmoother>> posSmoothers_;
    std::map<int, std::unique_ptr<EMASmoother>> poseSmoothers_;
    std::map<int, std::unique_ptr<MedianBuffer>> poseMedians_;
    std::map<int, std::vector<cv::Point2f>> lastCorners_;

    // Optical flow
    cv::Mat prevGray_;
    std::map<int, std::vector<cv::Point2f>> lkLastPts_;

    // Scene management
    std::chrono::steady_clock::time_point sceneUnseenStart_;
    bool sceneHasUnseen_;

    // Adaptive control
    double detectionRate_;
    std::deque<double> processTimeHist_;

    // EMA alphas
    double emaPosAlpha_;
    double emaPoseAlpha_;

    // Buffers (reused to avoid allocation)
    cv::Mat grayBuf_;
    cv::Mat preprocessBuf_;
};