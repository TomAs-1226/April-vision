#include "FrameProcessor.h"
#include "Config.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>

// Undefine Windows macros
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------- Safe SubPix helpers (local-only; no extra header needed) --------
namespace {
    // Returns true if any corner lies within 'margin' pixels of the border.
    inline bool anyCornerNearBorder(const std::vector<cv::Point2f>& c,
                                    int width, int height, int marginPx)
    {
        for (const auto& p : c) {
            if (p.x < marginPx || p.y < marginPx ||
                p.x >= width - marginPx || p.y >= height - marginPx)
                return true;
        }
        return false;
    }

    // Refines points in-place on 'src' but only for points whose sub-pix window
    // fits fully inside the image. This prevents the OpenCV assertion:
    // Rect(0,0,src.cols,src.rows).contains(cT) in cv::cornerSubPix   (imgproc/cornersubpix.cpp:99)
    // (OpenCV requires every refined point to be strictly inside the image.)  :contentReference[oaicite:2]{index=2}
    inline int safeCornerSubPix(cv::Mat src,
                                std::vector<cv::Point2f>& pts,
                                cv::Size winSize,
                                cv::Size zeroZone = {-1,-1},
                                cv::TermCriteria criteria = cv::TermCriteria(
                                    cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01))
    {
        if (src.empty() || pts.empty()) return 0;

        // cornerSubPix requires 8U/32F single-channel
        if (src.type() != CV_8UC1 && src.type() != CV_32FC1) {
            cv::Mat tmp;
            cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
            src = tmp;
        }

        const int rx = winSize.width  / 2;
        const int ry = winSize.height / 2;
        const cv::Rect safeROI(rx, ry,
                               std::max(0, src.cols - 2*rx),
                               std::max(0, src.rows - 2*ry));

        std::vector<cv::Point2f> safePts;
        safePts.reserve(pts.size());
        std::vector<int> idxMap; idxMap.reserve(pts.size());

        for (int i = 0; i < (int)pts.size(); ++i) {
            if (safeROI.contains(pts[i])) {
                safePts.push_back(pts[i]);
                idxMap.push_back(i);
            }
        }

        if (safePts.empty()) return 0;

        cv::cornerSubPix(src, safePts, winSize, zeroZone, criteria);

        for (int k = 0; k < (int)safePts.size(); ++k)
            pts[idxMap[k]] = safePts[k];

        return (int)safePts.size();
    }
} // namespace
// --------------------------------------------------------------------------


FrameProcessor::FrameProcessor()
    : tagSizeM_(config::DEFAULT_TAG_SIZE_M)
    , useCLAHE_(true)
    , gamma_(1.25)
    , baseDecimate_(config::DEFAULT_DECIMATE)
    , useWhitelist_(false)
    , useBlacklist_(false)
    , sceneHasUnseen_(false)
    , detectionRate_(config::DETECTION_RATE_HZ)
    , emaPosAlpha_(config::EMA_ALPHA_POS)
    , emaPoseAlpha_(config::EMA_ALPHA_POSE)
    , highSpeedMode_(config::DEFAULT_HIGH_SPEED_MODE)
    , highSpeedConfig_()
    , activeRoi_()
    , roiHoldFrames_(0)
    , fpsWindowStart_(std::chrono::steady_clock::now())
    , fpsWindowFrames_(0)
    , effectiveFps_(0.0)
{
    detector_ = std::make_unique<Detector>();
    poseEstimator_ = std::make_unique<PoseEstimator>();

    clahe_ = cv::createCLAHE(config::CLAHE_CLIP_LIMIT,
                             cv::Size(config::CLAHE_TILE_SIZE, config::CLAHE_TILE_SIZE));

    buildGammaLUT(gamma_);

    std::cout << "[FrameProcessor] Initialized" << std::endl;
}

FrameProcessor::~FrameProcessor() = default;

void FrameProcessor::resetTracking() {
    trackers_.clear();
    posSmoothers_.clear();
    poseSmoothers_.clear();
    poseMedians_.clear();
    lastCorners_.clear();
    lkLastPts_.clear();
    stableRvecs_.clear();
    stableTvecs_.clear();
    prevGray_.release();
    activeRoi_ = cv::Rect();
    roiHoldFrames_ = 0;
    sceneHasUnseen_ = false;
}

void FrameProcessor::setCameraMatrix(const cv::Mat& K, const cv::Mat& D) {
    K.copyTo(cameraMatrix_);
    D.copyTo(distCoeffs_);
}

void FrameProcessor::setEMAAlpha(double pos, double pose) {
    emaPosAlpha_ = pos;
    emaPoseAlpha_ = pose;
}

void FrameProcessor::setGamma(double gamma) {
    if (std::abs(gamma - gamma_) > 1e-6) {
        gamma_ = gamma;
        buildGammaLUT(gamma);
    }
}

void FrameProcessor::setHighSpeedMode(bool enabled) {
    if (highSpeedMode_ == enabled) {
        detector_->setRefineEdges(!highSpeedMode_);
        return;
    }
    highSpeedMode_ = enabled;
    if (!highSpeedMode_) {
        activeRoi_ = cv::Rect();
        roiHoldFrames_ = 0;
    }
    resetTracking();
    detector_->setRefineEdges(!highSpeedMode_);
}

void FrameProcessor::configureHighSpeed(const HighSpeedConfig& cfg) {
    highSpeedConfig_ = cfg;
    if (highSpeedConfig_.forcedSize.width > 0 && highSpeedConfig_.forcedSize.height > 0) {
        highSpeedConfig_.forcedSize.width = std::max(32, highSpeedConfig_.forcedSize.width);
        highSpeedConfig_.forcedSize.height = std::max(32, highSpeedConfig_.forcedSize.height);
    }
    highSpeedConfig_.roiPersistence = std::max(1, highSpeedConfig_.roiPersistence);
    highSpeedConfig_.minEdge = std::max(8, highSpeedConfig_.minEdge);
}

void FrameProcessor::buildGammaLUT(double gamma) {
    gammaLUT_.resize(256);
    if (std::abs(gamma - 1.0) < 1e-3) {
        for (int i = 0; i < 256; i++) gammaLUT_[i] = (uint8_t)i;
    } else {
        const double inv = 1.0 / gamma;
        for (int i = 0; i < 256; i++) {
            gammaLUT_[i] = static_cast<uint8_t>(cv::saturate_cast<uchar>(std::pow(i / 255.0, inv) * 255.0));
        }
    }
}

cv::Rect FrameProcessor::clampRectToImage(const cv::Rect& r, int width, int height) const {
    cv::Rect safe = r;
    safe.x = std::clamp(safe.x, 0, width);
    safe.y = std::clamp(safe.y, 0, height);
    safe.width = std::clamp(safe.width, 0, width - safe.x);
    safe.height = std::clamp(safe.height, 0, height - safe.y);
    return safe;
}

cv::Rect FrameProcessor::growRect(const cv::Rect& r, double scale, int width, int height, int minEdge) const {
    cv::Rect grown = r;
    if (grown.width == 0 || grown.height == 0) {
        grown = cv::Rect(0, 0, width, height);
    }
    const double cx = grown.x + grown.width / 2.0;
    const double cy = grown.y + grown.height / 2.0;
    double hw = std::max(static_cast<double>(grown.width), static_cast<double>(minEdge)) * scale * 0.5;
    double hh = std::max(static_cast<double>(grown.height), static_cast<double>(minEdge)) * scale * 0.5;
    cv::Rect2d expanded(cx - hw, cy - hh, hw * 2.0, hh * 2.0);
    cv::Rect integerRect(static_cast<int>(std::floor(expanded.x)),
                        static_cast<int>(std::floor(expanded.y)),
                        static_cast<int>(std::ceil(expanded.width)),
                        static_cast<int>(std::ceil(expanded.height)));
    return clampRectToImage(integerRect, width, height);
}

cv::Rect FrameProcessor::scaleRect(const cv::Rect& r, double sx, double sy) const {
    return cv::Rect(
        static_cast<int>(std::round(r.x * sx)),
        static_cast<int>(std::round(r.y * sy)),
        static_cast<int>(std::round(r.width * sx)),
        static_cast<int>(std::round(r.height * sy))
    );
}

cv::Vec3d FrameProcessor::rvecToEuler(const cv::Vec3d& rvec) const {
    if (cv::norm(rvec) < 1e-6) return {0, 0, 0};

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                          R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6;
    double roll, pitch, yaw;
    if (!singular) {
        roll  = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        pitch = std::atan2(-R.at<double>(2, 0), sy);
        yaw   = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        roll  = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        pitch = std::atan2(-R.at<double>(2, 0), sy);
        yaw   = 0;
    }

    const double rad2deg = 180.0 / M_PI;
    return {roll * rad2deg, pitch * rad2deg, yaw * rad2deg};
}

cv::Vec3d FrameProcessor::stabilizeRvec(int id, const cv::Vec3d& candidate) {
    auto it = stableRvecs_.find(id);
    if (it == stableRvecs_.end()) {
        stableRvecs_[id] = candidate;
        return candidate;
    }

    const cv::Vec3d& last = it->second;
    const cv::Vec3d flipped = -candidate;
    const double dDirect = cv::norm(candidate - last);
    const double dFlip = cv::norm(flipped - last);
    cv::Vec3d chosen = (dFlip < dDirect) ? flipped : candidate;

    const double alpha = 0.35; // heavier smoothing for axis stability
    cv::Vec3d blended = alpha * chosen + (1.0 - alpha) * last;
    stableRvecs_[id] = blended;
    return blended;
}

cv::Vec3d FrameProcessor::stabilizeTvec(int id, const cv::Vec3d& candidate) {
    auto it = stableTvecs_.find(id);
    if (it == stableTvecs_.end()) {
        stableTvecs_[id] = candidate;
        return candidate;
    }

    const cv::Vec3d& last = it->second;
    const double alpha = 0.30;
    cv::Vec3d blended = alpha * candidate + (1.0 - alpha) * last;
    stableTvecs_[id] = blended;
    return blended;
}

cv::Mat FrameProcessor::preprocessImage(const cv::Mat& gray) {
    if (preprocessBuf_.empty() || preprocessBuf_.size() != gray.size()) {
        preprocessBuf_.create(gray.size(), gray.type());
    }

    if (useCLAHE_) {
        clahe_->apply(gray, preprocessBuf_);
    } else {
        gray.copyTo(preprocessBuf_);
    }

    if (std::abs(gamma_ - 1.0) > 1e-3) {
        cv::LUT(preprocessBuf_, gammaLUT_, preprocessBuf_);
    }

    return preprocessBuf_;
}

double FrameProcessor::computeBlurVariance(const cv::Mat& gray) {
    cv::Mat lap; cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev; cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

int FrameProcessor::chooseDecimate(int base, double blurVar) {
    int decimate = std::max(1, base);
    if      (blurVar < config::BLUR_HIGH) decimate = std::min(decimate, config::ADAPT_DECIMATE_LOW);
    else if (blurVar < config::BLUR_MED)  decimate = std::min(decimate, config::ADAPT_DECIMATE_MED);
    else                                  decimate = std::min(decimate, config::ADAPT_DECIMATE_HIGH);
    return std::max(1, decimate);
}

bool FrameProcessor::shouldProcessTag(int id) {
    if (useWhitelist_ && whitelist_.find(id) == whitelist_.end()) return false;
    if (useBlacklist_ && blacklist_.find(id) != blacklist_.end()) return false;
    return true;
}

cv::Mat FrameProcessor::processFrame(const cv::Mat& frame, ProcessingStats& stats) {
    auto t0 = std::chrono::high_resolution_clock::now();
    stats = ProcessingStats{};

    if (frame.empty()) {
        return cv::Mat();
    }

    const int h = frame.rows;
    const int w = frame.cols;

    if (lastFrameSize_ != frame.size()) {
        resetTracking();
        lastFrameSize_ = frame.size();
    }

    // Initialize camera matrix if needed
    if (cameraMatrix_.empty()) {
        PoseEstimator::defaultCameraMatrix(w, h, cameraMatrix_, distCoeffs_);
    }

    if (grayBuf_.size() != frame.size()) {
        grayBuf_ = cv::Mat(h, w, CV_8UC1);
    }

    cv::Mat grayFull;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, grayBuf_, cv::COLOR_BGR2GRAY);
        grayFull = grayBuf_;
    } else {
        grayFull = frame;
    }

    // Track instantaneous FPS
    auto now = std::chrono::steady_clock::now();
    ++fpsWindowFrames_;
    double fpsWindowSeconds = std::chrono::duration<double>(now - fpsWindowStart_).count();
    if (fpsWindowSeconds >= 0.5) {
        effectiveFps_ = fpsWindowFrames_ / fpsWindowSeconds;
        fpsWindowFrames_ = 0;
        fpsWindowStart_ = now;
    }

    const double targetMs = 1000.0 / 60.0; // aim for >=60fps processing
    double recentAvgMs = 0.0;
    if (!processTimeHist_.empty()) {
        for (double t : processTimeHist_) recentAvgMs += t;
        recentAvgMs /= processTimeHist_.size();
    }

    cv::Mat working = grayFull;
    double scaleX = 1.0;
    double scaleY = 1.0;

    if (highSpeedMode_) {
        const bool needsResize = (highSpeedConfig_.forcedSize.width > 0 && highSpeedConfig_.forcedSize.height > 0 &&
                                 (grayFull.cols != highSpeedConfig_.forcedSize.width || grayFull.rows != highSpeedConfig_.forcedSize.height));
        if (needsResize) {
            if (resizedGray_.size() != highSpeedConfig_.forcedSize) {
                resizedGray_.create(highSpeedConfig_.forcedSize, CV_8UC1);
            }
            cv::resize(grayFull, resizedGray_, highSpeedConfig_.forcedSize, 0, 0, cv::INTER_AREA);
            working = resizedGray_;
            scaleX = static_cast<double>(grayFull.cols) / highSpeedConfig_.forcedSize.width;
            scaleY = static_cast<double>(grayFull.rows) / highSpeedConfig_.forcedSize.height;
        }
    } else {
        const double needScale = (recentAvgMs > targetMs)
            ? std::clamp(std::sqrt(targetMs / recentAvgMs), 0.65, 1.0)
            : 1.0;
        if (needScale < 0.999) {
            cv::Size scaledSize(
                std::max(1, static_cast<int>(grayFull.cols * needScale)),
                std::max(1, static_cast<int>(grayFull.rows * needScale)));
            if (resizedGray_.size() != scaledSize) {
                resizedGray_.create(scaledSize, CV_8UC1);
            }
            cv::resize(grayFull, resizedGray_, scaledSize, 0, 0, cv::INTER_AREA);
            working = resizedGray_;
            scaleX = static_cast<double>(grayFull.cols) / scaledSize.width;
            scaleY = static_cast<double>(grayFull.rows) / scaledSize.height;
        }

        activeRoi_ = cv::Rect();
        roiHoldFrames_ = 0;
    }

    cv::Mat grayProc = preprocessImage(working);
    if (!grayProc.isContinuous()) grayProc = grayProc.clone();

    cv::Rect roiRect(0, 0, grayProc.cols, grayProc.rows);
    bool roiActive = false;
    if (highSpeedMode_ && activeRoi_.area() > 0) {
        roiRect = clampRectToImage(activeRoi_, grayProc.cols, grayProc.rows);
        roiActive = roiRect.area() > 0 && roiRect.area() < grayProc.cols * grayProc.rows;
    }

    const double roiCoverage = roiActive
        ? static_cast<double>(roiRect.area()) / static_cast<double>(grayProc.cols * grayProc.rows)
        : 1.0;

    cv::Mat detectView = grayProc(roiRect);
    if (!detectView.isContinuous()) detectView = detectView.clone();

    const double blurVar = computeBlurVariance(detectView);
    int decimate = chooseDecimate(baseDecimate_, blurVar);
    if (recentAvgMs > targetMs * 1.5) {
        decimate = std::max(decimate, config::HIGH_SPEED_MIN_DECIMATE + 2);
    } else if (recentAvgMs > targetMs * 1.15) {
        decimate = std::max(decimate, config::HIGH_SPEED_MIN_DECIMATE + 1);
    }
    if (highSpeedMode_) {
        decimate = std::max(decimate, config::HIGH_SPEED_MIN_DECIMATE);
    }
    detector_->setDecimate(static_cast<float>(decimate));

    std::vector<Detection> filtered;
    std::vector<std::vector<cv::Point2f>> roiCornersList;
    auto detections = detector_->detect(detectView);
    filtered.reserve(detections.size());
    roiCornersList.reserve(detections.size());
    for (const auto& det : detections) {
        if (!shouldProcessTag(det.id)) continue;
        auto roiCorners = PoseEstimator::orderCorners(det.corners);
        for (auto& pt : roiCorners) {
            pt.x += roiRect.x;
            pt.y += roiRect.y;
        }

        std::vector<cv::Point2f> fullCorners = roiCorners;
        if (std::abs(scaleX - 1.0) > 1e-3 || std::abs(scaleY - 1.0) > 1e-3) {
            for (auto& pt : fullCorners) {
                pt.x = static_cast<float>(pt.x * scaleX);
                pt.y = static_cast<float>(pt.y * scaleY);
            }
        }

        Detection adjusted = det;
        adjusted.corners = fullCorners;
        filtered.push_back(std::move(adjusted));
        roiCornersList.push_back(std::move(roiCorners));
    }

    if (highSpeedMode_) {
        if (filtered.empty()) {
            if (roiHoldFrames_ > 0) {
                --roiHoldFrames_;
            } else {
                activeRoi_ = cv::Rect();
            }
        } else {
            roiHoldFrames_ = highSpeedConfig_.roiPersistence;
        }
    }

    cv::Mat vis;
    if (frame.channels() == 3) vis = frame.clone();
    else                       cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);

    std::set<int> visibleIds;
    std::vector<TagData> tagDataList;

    for (size_t i = 0; i < filtered.size(); ++i) {
        auto& det = filtered[i];
        const auto& roiCorners = roiCornersList[i];
        visibleIds.insert(det.id);
        std::vector<cv::Point2f> corners = PoseEstimator::orderCorners(det.corners);
        const double area = PoseEstimator::polygonArea(corners);

        if (!highSpeedMode_ && area > 30.0) {
            const cv::Size subWin(config::CORNER_SUBPIX_WIN, config::CORNER_SUBPIX_WIN);
            const int margin = std::max(subWin.width, subWin.height) / 2;
            if (!anyCornerNearBorder(corners, grayFull.cols, grayFull.rows, margin)) {
                safeCornerSubPix(grayFull, corners, subWin);
            }
        }

        if (highSpeedMode_) {
            cv::Rect detBox = cv::boundingRect(roiCorners);
            if (detBox.area() > 0) {
                detBox = clampRectToImage(detBox, grayProc.cols, grayProc.rows);
                cv::Rect grown = growRect(detBox, highSpeedConfig_.roiInflation, grayProc.cols, grayProc.rows, highSpeedConfig_.minEdge);
                if (activeRoi_.area() > 0) {
                    activeRoi_ = clampRectToImage(activeRoi_ | grown, grayProc.cols, grayProc.rows);
                } else {
                    activeRoi_ = grown;
                }
            }
        }

        updateTracker(det.id, corners);
        lastCorners_[det.id] = corners;

        double tx_deg, ty_deg, ta_percent;
        PoseEstimator::computeLimelightValues(corners, cameraMatrix_, w, h,
                                              tx_deg, ty_deg, ta_percent);

        cv::Vec3d tvec(0, 0, 0), rvec(0, 0, 0);
        cv::Vec3d rawT(0, 0, 0), rawR(0, 0, 0);
        double reprojErr = 0.0;

        if (cv::norm(corners[0] - corners[2]) >= 14.0 &&
            computePoseForCorners(det.id, corners, tvec, rvec, reprojErr, true, &rawT, &rawR)) {
            TagData td;
            td.id = det.id;
            td.tx_deg = tx_deg; td.ty_deg = ty_deg; td.ta_percent = ta_percent;
            td.tvec = tvec; td.rvec = rvec; td.reprojError = reprojErr;
            tagDataList.push_back(td);
        }

        const bool poseValid = cv::norm(tvec) > 0.0;
        if (poseValid && !stats.hasPose) {
            stats.hasPose = true;
            stats.poseTvec = tvec;
            stats.poseRvec = rvec;
            stats.poseEuler = rvecToEuler(rvec);
        }

        drawDetection(vis, det, corners, tx_deg, ty_deg, ta_percent, tvec, rvec, rawT, rawR, poseValid);
    }

    if (!prevGray_.empty()) {
        predictInvisibleTags(vis, w, h, visibleIds, prevGray_, grayFull, tagDataList, stats);
    }

    if (visibleIds.empty()) {
        if (!sceneHasUnseen_) {
            sceneUnseenStart_ = std::chrono::steady_clock::now();
            sceneHasUnseen_ = true;
        } else {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - sceneUnseenStart_).count();
            if (elapsed >= config::SCENE_PURGE_TIMEOUT) {
                trackers_.clear();
                posSmoothers_.clear();
                poseSmoothers_.clear();
                poseMedians_.clear();
                lastCorners_.clear();
                lkLastPts_.clear();
                sceneHasUnseen_ = false;
            }
        }
    } else {
        sceneHasUnseen_ = false;
    }

    purgeOldTrackers(visibleIds);
    grayFull.copyTo(prevGray_);

    if (publisher_ && !tagDataList.empty()) {
        VisionPayload payload;
        payload.timestamp = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        payload.tags = tagDataList;
        publisher_->publish(payload);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double procTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    processTimeHist_.push_back(procTimeMs);
    if (processTimeHist_.size() > 30) processTimeHist_.pop_front();

    if (processTimeHist_.size() == 30) {
        double avgMs = 0.0; for (double t : processTimeHist_) avgMs += t;
        avgMs /= processTimeHist_.size();
        const double minRate = static_cast<double>(config::MIN_DET_RATE);
        const double maxRate = static_cast<double>(config::MAX_DET_RATE);

        if (avgMs > config::PROCESS_TIME_HIGH_MS && detectionRate_ > minRate)
            detectionRate_ = std::max(minRate, detectionRate_ * 0.85);
        else if (avgMs < config::PROCESS_TIME_LOW_MS && detectionRate_ < maxRate)
            detectionRate_ = std::min(maxRate, detectionRate_ * 1.08);

        processTimeHist_.clear();
    }

    stats.detectionRateHz = detectionRate_;
    stats.avgProcessTimeMs = procTimeMs;
    stats.tagCount = static_cast<int>(filtered.size());
    stats.blurVariance = blurVar;
    stats.effectiveFps = effectiveFps_;
    stats.roiCoverage = roiCoverage;
    stats.roiActive = roiActive;
    stats.roiRect = roiActive ? scaleRect(roiRect, scaleX, scaleY) : cv::Rect();
    stats.highSpeedMode = highSpeedMode_;

    return vis;
}

void FrameProcessor::updateTracker(int id, const std::vector<cv::Point2f>& corners) {
    double cx = 0.0, cy = 0.0;
    for (const auto& pt : corners) { cx += pt.x; cy += pt.y; }
    cx /= corners.size(); cy /= corners.size();

    double diag = cv::norm(corners[0] - corners[2]);
    diag = std::max(config::MIN_SCALE_PX, diag);

    auto& tracker = trackers_[id];
    if (!tracker) {
        tracker = std::make_unique<BoxTracker>(config::TRACK_Q, config::TRACK_R);
        tracker->init(cx, cy, diag);
    } else {
        tracker->update(cx, cy, diag);
    }

    lkLastPts_[id] = corners;
}

bool FrameProcessor::computePoseForCorners(int id, const std::vector<cv::Point2f>& corners,
                                           cv::Vec3d& tvec, cv::Vec3d& rvec, double& reprojErr,
                                           bool smooth,
                                           cv::Vec3d* rawTvecOut, cv::Vec3d* rawRvecOut)
{
    auto poseResult = poseEstimator_->solvePose(corners, tagSizeM_, cameraMatrix_, distCoeffs_);
    if (!poseResult) {
        return false;
    }

    cv::Vec3d rawT(poseResult->tvec[0], poseResult->tvec[1], poseResult->tvec[2]);
    cv::Vec3d rawR(poseResult->rvec[0], poseResult->rvec[1], poseResult->rvec[2]);
    reprojErr = poseResult->reprojError;

    cv::Vec3d stableRawT = stabilizeTvec(id, rawT);
    cv::Vec3d stableRawR = stabilizeRvec(id, rawR);

    if (rawTvecOut) *rawTvecOut = stableRawT;
    if (rawRvecOut) *rawRvecOut = stableRawR;

    if (!smooth) {
        tvec = stableRawT;
        rvec = stableRawR;
        return true;
    }

    auto& posSmooth = posSmoothers_[id];
    if (!posSmooth || std::abs(posSmooth->getAlpha() - emaPosAlpha_) > 1e-9)
        posSmooth = std::make_unique<EMASmoother>(emaPosAlpha_);
    Eigen::Vector3d tvecE(rawT[0], rawT[1], rawT[2]);
    Eigen::VectorXd sPos = posSmooth->update(tvecE);

    auto& poseSmooth = poseSmoothers_[id];
    if (!poseSmooth || std::abs(poseSmooth->getAlpha() - emaPoseAlpha_) > 1e-9)
        poseSmooth = std::make_unique<EMASmoother>(emaPoseAlpha_);
    Eigen::Vector3d rvecE(rawR[0], rawR[1], rawR[2]);
    Eigen::VectorXd sRot = poseSmooth->update(rvecE);

    if (config::POSE_MEDIAN_WINDOW > 1) {
        auto& med = poseMedians_[id];
        if (!med) med = std::make_unique<MedianBuffer>(config::POSE_MEDIAN_WINDOW);
        Eigen::VectorXd both(6); both << sPos, sRot;
        med->push(both);
        if (auto m = med->median()) {
            tvec = cv::Vec3d((*m)(0), (*m)(1), (*m)(2));
            rvec = cv::Vec3d((*m)(3), (*m)(4), (*m)(5));
        } else {
            tvec = cv::Vec3d(sPos(0), sPos(1), sPos(2));
            rvec = cv::Vec3d(sRot(0), sRot(1), sRot(2));
        }
    } else {
        tvec = cv::Vec3d(sPos(0), sPos(1), sPos(2));
        rvec = cv::Vec3d(sRot(0), sRot(1), sRot(2));
    }

    tvec = stabilizeTvec(id, tvec);
    rvec = stabilizeRvec(id, rvec);

    return true;
}

void FrameProcessor::predictInvisibleTags(cv::Mat& vis, int width, int height,
                                         const std::set<int>& visibleIds,
                                         const cv::Mat& grayPrev, const cv::Mat& grayCurr,
                                         std::vector<TagData>& tagDataOut, ProcessingStats& stats)
{
    for (auto it = trackers_.begin(); it != trackers_.end(); ++it) {
        int id = it->first;
        auto& tracker = it->second;

        if (visibleIds.find(id) != visibleIds.end()) continue;

        double unseenTime = tracker->secondsUnseen();

        // Remove if too old
        if (unseenTime > config::REMOVAL_TIMEOUT) continue;

        // Skip if in fade-out period
        if (unseenTime > config::KEEPALIVE_TIMEOUT) {
            tracker->predict();
            continue;
        }

        // Try optical flow
        bool drewOpticalFlow = false;
        auto lkIt = lkLastPts_.find(id);

        if (lkIt != lkLastPts_.end() && !lkIt->second.empty()) {
            std::vector<cv::Point2f> p1;
            std::vector<uchar> status;
            std::vector<float> err;

            cv::calcOpticalFlowPyrLK(
                grayPrev, grayCurr, lkIt->second, p1, status, err,
                cv::Size(config::LK_WIN_SIZE, config::LK_WIN_SIZE),
                config::LK_MAX_LEVEL,
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                 config::LK_MAX_ITER, config::LK_EPSILON)
            );

            int goodCount = 0;
            for (size_t i = 0; i < status.size(); i++) if (status[i]) goodCount++;

            if (goodCount >= 3 && p1.size() >= 4) {
                // Clamp to image bounds defensively
                for (auto& pt : p1) {
                    pt.x = std::max(1.0f, std::min(pt.x, static_cast<float>(width  - 2)));
                    pt.y = std::max(1.0f, std::min(pt.y, static_cast<float>(height - 2)));
                }

                // Update tracker with OF result
                double cx = 0.0, cy = 0.0;
                for (const auto& pt : p1) { cx += pt.x; cy += pt.y; }
                cx /= p1.size(); cy /= p1.size();

                double diag = cv::norm(p1[0] - p1[2]);
                diag = std::clamp(diag, config::MIN_SCALE_PX, config::MAX_SCALE_PX);

                tracker->update(cx, cy, diag);
                lkLastPts_[id] = p1;
                lastCorners_[id] = p1;

                // Draw OF prediction
                std::vector<cv::Point> pts;
                pts.reserve(p1.size());
                for (const auto& pt : p1)
                    pts.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
                cv::polylines(vis, pts, true, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);

                cv::Vec3d tvec(0, 0, 0), rvec(0, 0, 0);
                cv::Vec3d rawT(0, 0, 0), rawR(0, 0, 0);
                double reprojErr = 0.0;
                if (cameraMatrix_.total() > 0 && computePoseForCorners(id, p1, tvec, rvec, reprojErr, true, &rawT, &rawR)) {
                    double tx_deg = 0.0, ty_deg = 0.0, ta = 0.0;
                    PoseEstimator::computeLimelightValues(p1, cameraMatrix_, width, height, tx_deg, ty_deg, ta);

                    TagData td;
                    td.id = id;
                    td.tx_deg = tx_deg; td.ty_deg = ty_deg; td.ta_percent = ta;
                    td.tvec = tvec; td.rvec = rvec; td.reprojError = reprojErr;
                    tagDataOut.push_back(td);

                    if (!stats.hasPose) {
                        stats.hasPose = true;
                        stats.poseTvec = tvec;
                        stats.poseRvec = rvec;
                        stats.poseEuler = rvecToEuler(rvec);
                    }

                    cv::drawFrameAxes(vis, cameraMatrix_, distCoeffs_, rawR, rawT,
                                      static_cast<float>(tagSizeM_), 2);
                }

                drewOpticalFlow = true;
            }
        }

        // Fall back to Kalman-only prediction
        if (!drewOpticalFlow) {
            tracker->predict();
            if (auto state = tracker->get()) {
                auto [cx, cy, s] = *state;

                // Limit prediction distance vs last known
                auto lastIt = lastCorners_.find(id);
                if (lastIt != lastCorners_.end()) {
                    double lastCx = 0.0, lastCy = 0.0;
                    for (const auto& pt : lastIt->second) { lastCx += pt.x; lastCy += pt.y; }
                    lastCx /= lastIt->second.size(); lastCy /= lastIt->second.size();

                    const double dist = std::hypot(cx - lastCx, cy - lastCy);
                    const double maxMove = config::MAX_PREDICT_DISTANCE * std::hypot(width, height);
                    if (dist > maxMove) {
                        const double f = maxMove / (dist + 1e-6);
                        cx = lastCx + (cx - lastCx) * f;
                        cy = lastCy + (cy - lastCy) * f;
                    }
                }

                // Clamp to extended bounds
                const int pad = std::min(width, height) / 2;
                cx = std::clamp(cx, -static_cast<double>(pad), static_cast<double>(width  - 1 + pad));
                cy = std::clamp(cy, -static_cast<double>(pad), static_cast<double>(height - 1 + pad));
                s  = std::clamp(s,  config::MIN_SCALE_PX,       config::MAX_SCALE_PX);

                drawPrediction(vis, id, cx, cy, s, false);

                std::vector<cv::Point2f> predictedCorners;
                if (cameraMatrix_.total() > 0 && buildPredictedCorners(id, cx, cy, s, predictedCorners)) {
                    double tx_deg = 0.0, ty_deg = 0.0, ta = 0.0;
                    PoseEstimator::computeLimelightValues(predictedCorners, cameraMatrix_, width, height, tx_deg, ty_deg, ta);

                    cv::Vec3d tvec(0, 0, 0), rvec(0, 0, 0);
                    cv::Vec3d rawT(0, 0, 0), rawR(0, 0, 0);
                    double reprojErr = 0.0;
                    if (computePoseForCorners(id, predictedCorners, tvec, rvec, reprojErr, true, &rawT, &rawR)) {
                        TagData td;
                        td.id = id;
                        td.tx_deg = tx_deg; td.ty_deg = ty_deg; td.ta_percent = ta;
                        td.tvec = tvec; td.rvec = rvec; td.reprojError = reprojErr;
                        tagDataOut.push_back(td);

                        if (!stats.hasPose) {
                            stats.hasPose = true;
                            stats.poseTvec = tvec;
                            stats.poseRvec = rvec;
                            stats.poseEuler = rvecToEuler(rvec);
                        }

                        cv::drawFrameAxes(vis, cameraMatrix_, distCoeffs_, rawR, rawT,
                                          static_cast<float>(tagSizeM_), 2);
                    }
                }
            }
        }
    }
}

bool FrameProcessor::buildPredictedCorners(int id, double cx, double cy, double scale,
                                           std::vector<cv::Point2f>& out) {
    auto last = lastCorners_.find(id);
    if (last == lastCorners_.end() || last->second.size() != 4) return false;

    const auto ordered = PoseEstimator::orderCorners(last->second);
    double baseCx = 0.0, baseCy = 0.0;
    for (const auto& pt : ordered) { baseCx += pt.x; baseCy += pt.y; }
    baseCx /= ordered.size(); baseCy /= ordered.size();

    const double baseDiag = std::max(1.0, cv::norm(ordered[0] - ordered[2]));
    const double s = scale / baseDiag;

    out.resize(4);
    for (size_t i = 0; i < ordered.size(); ++i) {
        const double dx = ordered[i].x - baseCx;
        const double dy = ordered[i].y - baseCy;
        out[i] = cv::Point2f(static_cast<float>(cx + dx * s),
                             static_cast<float>(cy + dy * s));
    }
    return true;
}

void FrameProcessor::purgeOldTrackers(const std::set<int>& visibleIds) {
    std::vector<int> toRemove;
    for (const auto& [id, tracker] : trackers_) {
        if (visibleIds.find(id) == visibleIds.end()) {
            if (tracker->secondsUnseen() > config::REMOVAL_TIMEOUT) {
                toRemove.push_back(id);
            }
        }
    }
    for (int id : toRemove) {
        trackers_.erase(id);
        posSmoothers_.erase(id);
        poseSmoothers_.erase(id);
        poseMedians_.erase(id);
        lastCorners_.erase(id);
        lkLastPts_.erase(id);
        stableRvecs_.erase(id);
        stableTvecs_.erase(id);
    }
}

void FrameProcessor::drawDetection(cv::Mat& vis, const Detection& det,
                                  const std::vector<cv::Point2f>& corners,
                                  double tx, double ty, double ta,
                                  const cv::Vec3d& tvec, const cv::Vec3d& rvec,
                                  const cv::Vec3d& rawTvec, const cv::Vec3d& rawRvec,
                                  bool poseValid)
{
    std::vector<cv::Point> pts; pts.reserve(corners.size());
    for (const auto& pt : corners)
        pts.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
    cv::polylines(vis, pts, true, cv::Scalar(0, 220, 0), 2, cv::LINE_AA);

    double cx = 0.0, cy = 0.0;
    for (const auto& pt : corners) { cx += pt.x; cy += pt.y; }
    cx /= corners.size(); cy /= corners.size();

    std::ostringstream oss;
    oss << "ID" << det.id
        << " tx=" << std::fixed << std::setprecision(1) << tx
        << "° ty=" << ty
        << "° ta=" << std::setprecision(2) << ta << "%";
    if (poseValid) {
        oss << " Z="  << std::setprecision(3) << tvec[2] << "m";
    }

    std::string text = oss.str();
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 2, &baseline);

    int x0 = std::max(2, static_cast<int>(cx - textSize.width / 2));
    int y0 = std::max(2, static_cast<int>(cy - textSize.height - 8));
    x0 = std::min(x0, vis.cols - textSize.width - 2);
    y0 = std::min(y0, vis.rows - textSize.height - 2);

    cv::rectangle(vis, {x0 - 4, y0 - 4}, {x0 + textSize.width + 4, y0 + textSize.height + 6},
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(vis, text, {x0, y0 + textSize.height},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    const std::string idText = "id=" + std::to_string(det.id);
    int tx_text = static_cast<int>(corners[0].x);
    int ty_text = static_cast<int>(corners[0].y - 20);
    const cv::Size idSize = cv::getTextSize(idText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    cv::rectangle(vis,
                  {tx_text - 3, ty_text - idSize.height - 3},
                  {tx_text + idSize.width + 3, ty_text + 3},
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(vis, idText, {tx_text, ty_text},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    if (poseValid && !cameraMatrix_.empty() && cameraMatrix_.total() > 0) {
        const cv::Vec3d& axisR = (cv::norm(rawRvec) > 0.0) ? rawRvec : rvec;
        const cv::Vec3d& axisT = (cv::norm(rawTvec) > 0.0) ? rawTvec : tvec;
        cv::drawFrameAxes(vis, cameraMatrix_, distCoeffs_, axisR, axisT,
                          static_cast<float>(tagSizeM_), 2);
    }
}

void FrameProcessor::drawPrediction(cv::Mat& vis, int id, double cx, double cy, double s, bool isOpticalFlow) {
    const int half = std::max(6, static_cast<int>(s * 0.5));
    const cv::Point p1(static_cast<int>(cx - half), static_cast<int>(cy - half));
    const cv::Point p2(static_cast<int>(cx + half), static_cast<int>(cy + half));
    const cv::Scalar color = isOpticalFlow ? cv::Scalar(0, 180, 255) : cv::Scalar(0, 160, 255);
    cv::rectangle(vis, p1, p2, color, 2, cv::LINE_AA);
}
