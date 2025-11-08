#include "PoseEstimator.h"
#include "Config.h"
#include <cmath>
#include <algorithm>

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

    // 3D object points (tag corners in tag frame)
    const double half = tagSizeMeters / 2.0;
    std::vector<cv::Point3f> objPoints = {
        cv::Point3f(static_cast<float>(-half), static_cast<float>(-half), 0.0f),
        cv::Point3f(static_cast<float>(half), static_cast<float>(-half), 0.0f),
        cv::Point3f(static_cast<float>(half), static_cast<float>(half), 0.0f),
        cv::Point3f(static_cast<float>(-half), static_cast<float>(half), 0.0f)
    };

    cv::Vec3d rvec, tvec;

    // Try IPPE_SQUARE first
    bool success = false;
    try {
        success = cv::solvePnP(objPoints, corners, cameraMatrix, distCoeffs,
                              rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
    } catch (...) {
        success = false;
    }

    // Fallback to iterative
    if (!success) {
        try {
            success = cv::solvePnP(objPoints, corners, cameraMatrix, distCoeffs,
                                  rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        } catch (...) {
            return std::nullopt;
        }
    }

    if (!success) {
        return std::nullopt;
    }

    // Compute reprojection error
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(objPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    double totalError = 0.0;
    for (size_t i = 0; i < corners.size(); i++) {
        cv::Point2f diff = corners[i] - projectedPoints[i];
        totalError += cv::norm(diff);
    }
    double avgError = totalError / corners.size();

    // Validate reprojection error
    if (!std::isfinite(avgError) || avgError > config::REPROJ_ERR_THRESH) {
        return std::nullopt;
    }

    PoseResult result;
    result.rvec = rvec;
    result.tvec = tvec;
    result.reprojError = avgError;

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