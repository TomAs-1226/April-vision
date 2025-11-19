#include "FieldLayout.h"

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <iostream>
#include <cmath>

bool FieldLayout::loadFromJson(const std::string& path, bool clearExisting) {
    if (clearExisting) {
        poses_.clear();
    }

    const size_t before = poses_.size();

    cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) {
        std::cerr << "[FieldLayout] Failed to open layout: " << path << std::endl;
        return false;
    }

    cv::FileNode tagsNode = fs["tags"];
    if (tagsNode.type() != cv::FileNode::SEQ) {
        std::cerr << "[FieldLayout] Layout missing 'tags' array" << std::endl;
        return false;
    }

    for (const auto& tagNode : tagsNode) {
        if (!tagNode.isMap()) continue;
        FieldTagPose pose{};
        pose.id = static_cast<int>(tagNode["ID"]);

        cv::FileNode poseNode = tagNode["pose"];
        cv::FileNode translation = poseNode["translation"];
        cv::FileNode rotation = poseNode["rotation"]["quaternion"];
        if (translation.empty() || rotation.empty()) {
            continue;
        }

        pose.translation = cv::Vec3d(
            static_cast<double>(translation["x"]),
            static_cast<double>(translation["y"]),
            static_cast<double>(translation["z"]) );

        const double w = static_cast<double>(rotation["W"]);
        const double x = static_cast<double>(rotation["X"]);
        const double y = static_cast<double>(rotation["Y"]);
        const double z = static_cast<double>(rotation["Z"]);
        pose.rotation = quaternionToMatrix(w, x, y, z);

        poses_[pose.id] = pose;
    }

    const size_t loaded = poses_.size() - before;
    std::cout << "[FieldLayout] Loaded " << loaded
              << " tag poses from " << path << std::endl;
    return !poses_.empty();
}

std::optional<FieldTagPose> FieldLayout::getTagPose(int id) const {
    auto it = poses_.find(id);
    if (it == poses_.end()) {
        return std::nullopt;
    }
    return it->second;
}

cv::Matx33d FieldLayout::quaternionToMatrix(double w, double x, double y, double z) {
    const double norm = std::sqrt(w*w + x*x + y*y + z*z);
    const double inv = norm > 1e-9 ? 1.0 / norm : 0.0;
    w *= inv; x *= inv; y *= inv; z *= inv;

    const double xx = x * x;
    const double yy = y * y;
    const double zz = z * z;
    const double xy = x * y;
    const double xz = x * z;
    const double yz = y * z;
    const double wx = w * x;
    const double wy = w * y;
    const double wz = w * z;

    return cv::Matx33d(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)
    );
}
