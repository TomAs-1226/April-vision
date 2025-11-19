#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <unordered_map>

struct FieldTagPose {
    int id;
    cv::Vec3d translation;   // tag origin in field frame (meters)
    cv::Matx33d rotation;    // rotation from tag frame to field frame
};

class FieldLayout {
public:
    bool loadFromJson(const std::string& path);
    std::optional<FieldTagPose> getTagPose(int id) const;
    bool empty() const { return poses_.empty(); }

private:
    static cv::Matx33d quaternionToMatrix(double w, double x, double y, double z);
    std::unordered_map<int, FieldTagPose> poses_;
};
