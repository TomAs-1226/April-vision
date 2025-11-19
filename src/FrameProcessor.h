#pragma once

#include "Detector.h"
#include "PoseEstimator.h"
#include "ArucoPoseHelper.h"
#include "Tracker.h"
#include "NetworkPublisher.h"
#include "FieldLayout.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <memory>
#include <map>
#include <set>
#include <deque>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <optional>

struct ProcessingStats {
    double detectionRateHz;
    double avgProcessTimeMs;
    int tagCount;
    double blurVariance;
};

struct FilterOutput {
    cv::Vec3d filteredTranslation;
    cv::Vec3d predictedTranslation;
    cv::Vec3d velocity;
    cv::Vec3d acceleration;
    cv::Vec3d filteredRvec;
    cv::Vec3d filteredRpyDeg;
    bool usedMeasurement = false;
    bool predictionOnly = false;
};

struct PoseFilterState {
    cv::Vec3d position{0.0, 0.0, 0.0};
    cv::Vec3d velocity{0.0, 0.0, 0.0};
    cv::Vec3d acceleration{0.0, 0.0, 0.0};
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    bool initialized = false;
    bool orientationInitialized = false;
    std::chrono::steady_clock::time_point lastUpdate{};
    std::chrono::steady_clock::time_point lastMeasurement{};
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
    void applyTemporalDenoise(cv::Mat& img);
    void applySpecularSuppression(cv::Mat& img);
    double computeGlareFraction(const std::vector<cv::Point2f>& corners) const;

    // Detection filtering
    bool shouldProcessTag(int id);
    std::vector<Detection> detectWithStrategy(const cv::Mat& grayProc);
    std::vector<cv::Rect> buildDynamicRois(int width, int height) const;
    std::vector<Detection> mergeDetectionsById(const std::vector<Detection>& dets) const;

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
    std::optional<TargetSummary> selectBestTarget(const std::vector<TagData>& tags,
                                                 int width, int height, int& bestIndex);
    std::optional<MultiTagSolution> solveFieldPose(const std::vector<TagData>& tags);
    double updateDistanceHistory(int id, double distanceM);
    void updateFastMode(double procTimeMs);
    void applyPoseFilters(std::vector<TagData>& tags,
                          double pipelineLatencyMs,
                          const std::set<int>& visibleIds,
                          std::chrono::steady_clock::time_point frameTime);
    FilterOutput stepPoseFilter(int id,
                                const std::optional<cv::Vec3d>& translation,
                                const std::optional<cv::Vec3d>& rvec,
                                double latencySec,
                                std::chrono::steady_clock::time_point timestamp);
    std::vector<TagData> emitPredictedTags(const std::set<int>& visibleIds,
                                           double latencySec,
                                           double pipelineLatencyMs,
                                           std::chrono::steady_clock::time_point frameTime);
    void dropPoseFilter(int id);

    // Members
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<PoseEstimator> poseEstimator_;
    std::unique_ptr<ArucoPoseHelper> arucoHelper_;
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

    struct DistanceHistory {
        double distance = 0.0;
        double velocity = 0.0;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::unordered_map<int, DistanceHistory> distanceHistory_;
    std::unordered_map<int, PoseFilterState> poseFilters_;

    // Scene management
    std::chrono::steady_clock::time_point sceneUnseenStart_;
    bool sceneHasUnseen_;

    // Adaptive control
    double detectionRate_;
    std::deque<double> processTimeHist_;
    size_t frameCounter_ = 0;

    // EMA alphas
    double emaPosAlpha_;
    double emaPoseAlpha_;

    // Buffers (reused to avoid allocation)
    cv::Mat grayBuf_;
    cv::Mat preprocessBuf_;
    cv::Mat temporalLowpass_;
    cv::Mat glareMask_;
    bool glareSuppressedThisFrame_ = false;

    // Field layout / extrinsics
    std::unique_ptr<FieldLayout> fieldLayout_;
    bool fieldLayoutReady_ = false;
    cv::Matx33d camToRobotR_;
    cv::Matx33d robotToCamR_;
    cv::Vec3d camToRobotT_;
    cv::Vec3d robotToCamT_;

    // Fast-path scheduling
    int fastModeCounter_ = 0;
    bool fastModeActive_ = false;
};