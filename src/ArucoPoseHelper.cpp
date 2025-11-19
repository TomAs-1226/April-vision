#include "ArucoPoseHelper.h"
#include <opencv2/aruco.hpp>
#include <cmath>

ArucoPoseHelper::ArucoPoseHelper() {
    // Default square tag points centred at origin; scaled on demand
    objPoints_.resize(4);
}

bool ArucoPoseHelper::estimatePose(const std::vector<cv::Point2f>& corners,
                                   double tagSizeMeters,
                                   const cv::Mat& cameraMatrix,
                                   const cv::Mat& distCoeffs,
                                   cv::Vec3d& rvec,
                                   cv::Vec3d& tvec,
                                   double& reprojError)
{
    if (corners.size() != 4 || tagSizeMeters <= 0.0) {
        return false;
    }

    const double half = tagSizeMeters * 0.5;
    objPoints_[0] = cv::Point3f(static_cast<float>(-half), static_cast<float>(half), 0.0f);
    objPoints_[1] = cv::Point3f(static_cast<float>(half), static_cast<float>(half), 0.0f);
    objPoints_[2] = cv::Point3f(static_cast<float>(half), static_cast<float>(-half), 0.0f);
    objPoints_[3] = cv::Point3f(static_cast<float>(-half), static_cast<float>(-half), 0.0f);

    std::vector<std::vector<cv::Point2f>> markerCorners{corners};
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;

    try {
        cv::aruco::estimatePoseSingleMarkers(markerCorners,
                                             tagSizeMeters,
                                             cameraMatrix,
                                             distCoeffs,
                                             rvecs,
                                             tvecs);
    } catch (...) {
        return false;
    }

    if (rvecs.empty() || tvecs.empty()) {
        return false;
    }

    rvec = rvecs[0];
    tvec = tvecs[0];

    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(objPoints_, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    double totalError = 0.0;
    for (size_t i = 0; i < corners.size(); ++i) {
        const cv::Point2f diff = corners[i] - projectedPoints[i];
        totalError += cv::norm(diff);
    }
    reprojError = totalError / corners.size();

    return std::isfinite(reprojError);
}
