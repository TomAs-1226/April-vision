#include "PoseEstimator.h"
#include "Config.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

// Define M_PI for MSVC
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============ EMASmoother ============
EMASmoother::EMASmoother(double alpha)
    : alpha_(alpha), initialized_(false) {}

Eigen::VectorXd EMASmoother::update(const Eigen::VectorXd& val) {
    if (!initialized_) {
        v_ = val;
        initialized_ = true;
    } else {
        v_ = alpha_ * val + (1.0 - alpha_) * v_;
    }
    return v_;
}

void EMASmoother::reset() {
    initialized_ = false;
}

// ============ MedianBuffer ============
MedianBuffer::MedianBuffer(int window)
    : window_(std::max(1, window)) {}

void MedianBuffer::push(const Eigen::VectorXd& val) {
    q_.push_back(val);
    if (q_.size() > static_cast<size_t>(window_)) {
        q_.pop_front();
    }
}

std::optional<Eigen::VectorXd> MedianBuffer::median() {
    if (q_.empty()) {
        return std::nullopt;
    }

    const int dim = static_cast<int>(q_.front().size());
    Eigen::VectorXd result(dim);

    std::vector<double> values;
    values.reserve(q_.size());

    for (int d = 0; d < dim; d++) {
        values.clear();
        for (const auto& v : q_) {
            values.push_back(v(d));
        }

        std::nth_element(values.begin(),
                        values.begin() + values.size() / 2,
                        values.end());
        result(d) = values[values.size() / 2];
    }

    return result;
}

void MedianBuffer::clear() {
    q_.clear();
}

// ============ PoseEstimator ============
PoseEstimator::PoseEstimator() {}

std::optional<PoseResult> PoseEstimator::solvePose(
    const std::vector<cv::Point2f>& corners,
    double tagSizeMeters,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs)
{
    if (corners.size() != 4) {
        return std::nullopt;
    }

    const std::vector<cv::Point2f> orderedCorners = orderCorners(corners);

    // 3D object points (tag corners in tag frame)
    const double half = tagSizeMeters / 2.0;
    std::vector<cv::Point3f> objPoints = {
        cv::Point3f(static_cast<float>(-half), static_cast<float>(half), 0.0f),   // top-left
        cv::Point3f(static_cast<float>(half), static_cast<float>(half), 0.0f),    // top-right
        cv::Point3f(static_cast<float>(half), static_cast<float>(-half), 0.0f),   // bottom-right
        cv::Point3f(static_cast<float>(-half), static_cast<float>(-half), 0.0f)   // bottom-left
    };

    cv::Vec3d rvec, tvec;

    auto solveAndScore = [&](int method, cv::Vec3d& outR, cv::Vec3d& outT) -> std::optional<double> {
        try {
            if (!cv::solvePnP(objPoints, orderedCorners, cameraMatrix, distCoeffs,
                               outR, outT, false, method)) {
                return std::nullopt;
            }
        } catch (...) {
            return std::nullopt;
        }

        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objPoints, outR, outT, cameraMatrix, distCoeffs, projectedPoints);
        double totalError = 0.0;
        for (size_t i = 0; i < orderedCorners.size(); i++) {
            cv::Point2f diff = orderedCorners[i] - projectedPoints[i];
            totalError += cv::norm(diff);
        }
        const double avgError = totalError / orderedCorners.size();
        if (!std::isfinite(avgError)) return std::nullopt;
        return avgError;
    };

    cv::Vec3d bestR, bestT;
    double bestErr = std::numeric_limits<double>::max();

    const int methods[] = {cv::SOLVEPNP_IPPE_SQUARE, cv::SOLVEPNP_SQPNP, cv::SOLVEPNP_ITERATIVE};
    for (int method : methods) {
        cv::Vec3d rTmp, tTmp;
        if (auto err = solveAndScore(method, rTmp, tTmp)) {
            if (*err < bestErr) {
                bestErr = *err;
                bestR = rTmp;
                bestT = tTmp;
            }
        }
    }

    if (!std::isfinite(bestErr) || bestErr > config::REPROJ_ERR_THRESH) {
        return std::nullopt;
    }

    try {
        cv::solvePnPRefineLM(objPoints, orderedCorners, cameraMatrix, distCoeffs, bestR, bestT);
        if (auto err = solveAndScore(cv::SOLVEPNP_ITERATIVE, bestR, bestT)) {
            bestErr = *err;
        }
    } catch (...) {
        // keep previous best
    }

    PoseResult result;
    result.rvec = bestR;
    result.tvec = bestT;
    result.reprojError = bestErr;

    return result;
}

void PoseEstimator::defaultCameraMatrix(int width, int height, cv::Mat& K, cv::Mat& D) {
    // FIXED: Avoid std::max macro issue - compute inline
    int maxDim = (width > height) ? width : height;
    double focalLength = 0.9 * static_cast<double>(maxDim);

    K = (cv::Mat_<double>(3, 3) <<
        focalLength,   0.0,          static_cast<double>(width) / 2.0,
        0.0,           focalLength,  static_cast<double>(height) / 2.0,
        0.0,           0.0,          1.0
    );

    D = cv::Mat::zeros(5, 1, CV_64F);
}

void PoseEstimator::computeLimelightValues(
    const std::vector<cv::Point2f>& corners,
    const cv::Mat& K,
    int width, int height,
    double& tx_deg, double& ty_deg, double& ta_percent)
{
    // Compute centroid
    double cx = 0.0, cy = 0.0;
    for (const auto& pt : corners) {
        cx += pt.x;
        cy += pt.y;
    }
    cx /= corners.size();
    cy /= corners.size();

    // Extract camera parameters
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx0 = K.at<double>(0, 2);
    double cy0 = K.at<double>(1, 2);

    // Compute angles
    double tx_rad = std::atan2(cx - cx0, fx);
    double ty_rad = std::atan2(cy0 - cy, fy);

    tx_deg = tx_rad * 180.0 / M_PI;
    ty_deg = ty_rad * 180.0 / M_PI;

    // Compute area percentage
    double area_px = polygonArea(corners);
    ta_percent = 100.0 * area_px / (width * height);
}

double PoseEstimator::polygonArea(const std::vector<cv::Point2f>& pts) {
    if (pts.size() < 3) return 0.0;

    double area = 0.0;
    for (size_t i = 0; i < pts.size(); i++) {
        size_t j = (i + 1) % pts.size();
        area += pts[i].x * pts[j].y;
        area -= pts[j].x * pts[i].y;
    }
    return std::abs(area) * 0.5;
}

std::vector<cv::Point2f> PoseEstimator::orderCorners(const std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4) {
        return corners;
    }

    cv::Point2f center(0.f, 0.f);
    for (const auto& pt : corners) {
        center += pt;
    }
    center *= 0.25f;

    std::vector<std::pair<double, cv::Point2f>> anglePoints;
    anglePoints.reserve(4);
    for (const auto& pt : corners) {
        double angle = std::atan2(pt.y - center.y, pt.x - center.x);
        anglePoints.emplace_back(angle, pt);
    }

    std::sort(anglePoints.begin(), anglePoints.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::vector<cv::Point2f> ordered;
    ordered.reserve(4);
    for (const auto& ap : anglePoints) {
        ordered.push_back(ap.second);
    }

    auto topLeftIt = std::min_element(ordered.begin(), ordered.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        const float sumA = a.x + a.y;
        const float sumB = b.x + b.y;
        if (std::abs(sumA - sumB) < 1e-4f) {
            if (std::abs(a.y - b.y) < 1e-4f) {
                return a.x < b.x;
            }
            return a.y < b.y;
        }
        return sumA < sumB;
    });

    if (topLeftIt != ordered.begin()) {
        std::rotate(ordered.begin(), topLeftIt, ordered.end());
    }

    const double cross =
        (ordered[1].x - ordered[0].x) * (ordered[2].y - ordered[0].y) -
        (ordered[1].y - ordered[0].y) * (ordered[2].x - ordered[0].x);
    if (cross < 0.0) {
        std::reverse(ordered.begin() + 1, ordered.end());
    }

    return ordered;
}