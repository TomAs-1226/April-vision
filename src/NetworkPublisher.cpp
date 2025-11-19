#include "NetworkPublisher.h"
#include "Config.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <limits>
#include <cmath>
#include <algorithm>
#ifndef _WIN32
#include <fcntl.h>
#endif

#include <array>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
constexpr double DEG2RAD = M_PI / 180.0;
constexpr double RAD2DEG = 180.0 / M_PI;

cv::Matx33d rpyDegToMatrix(double rollDeg, double pitchDeg, double yawDeg) {
    const double r = rollDeg * DEG2RAD;
    const double p = pitchDeg * DEG2RAD;
    const double y = yawDeg * DEG2RAD;

    const double c1 = std::cos(y);
    const double s1 = std::sin(y);
    const double c2 = std::cos(p);
    const double s2 = std::sin(p);
    const double c3 = std::cos(r);
    const double s3 = std::sin(r);

    cv::Matx33d Rz(c1, -s1, 0,
                   s1,  c1, 0,
                     0,   0, 1);
    cv::Matx33d Ry(c2, 0, s2,
                     0, 1,  0,
                  -s2, 0, c2);
    cv::Matx33d Rx(1,  0,  0,
                   0, c3, -s3,
                   0, s3,  c3);

    return Rz * Ry * Rx;
}

cv::Matx33d matFromCv(const cv::Mat& mat) {
    cv::Matx33d out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = mat.at<double>(r, c);
        }
    }
    return out;
}

cv::Vec3d matMul(const cv::Matx33d& R, const cv::Vec3d& v) {
    return cv::Vec3d(
        R(0,0) * v[0] + R(0,1) * v[1] + R(0,2) * v[2],
        R(1,0) * v[0] + R(1,1) * v[1] + R(1,2) * v[2],
        R(2,0) * v[0] + R(2,1) * v[1] + R(2,2) * v[2]
    );
}

std::array<double, 3> matrixToEulerDeg(const cv::Matx33d& R) {
    const double roll = std::atan2(R(2, 1), R(2, 2)) * RAD2DEG;
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0)) * RAD2DEG;
    const double yaw = std::atan2(R(1, 0), R(0, 0)) * RAD2DEG;
    return {roll, pitch, yaw};
}

std::vector<double> poseVector(const cv::Vec3d& t, const cv::Matx33d& R) {
    const auto euler = matrixToEulerDeg(R);
    return {t[0], t[1], t[2], euler[0], euler[1], euler[2]};
}

std::pair<cv::Vec3d, cv::Matx33d> mirrorBlueToRed(const cv::Vec3d& posBlue,
                                                  const cv::Matx33d& rotBlue) {
    cv::Vec3d posRed(
        config::FIELD_LENGTH_METERS - posBlue[0],
        config::FIELD_WIDTH_METERS - posBlue[1],
        posBlue[2]
    );
    static const cv::Matx33d flip(
        -1, 0, 0,
         0,-1, 0,
         0, 0, 1
    );
    cv::Matx33d rotRed = flip * rotBlue;
    return {posRed, rotRed};
}
} // namespace

NetworkPublisher::NetworkPublisher(const std::string& ntServer,
                                 const std::string& udpIp,
                                 int udpPort)
    : ntServer_(ntServer)
    , udpIp_(udpIp)
    , udpPort_(udpPort)
    , running_(false)
    , ntEnabled_(false)
    , udpEnabled_(true)
    , udpSocket_(-1)
    , udpInitialized_(false)
{
#ifdef USE_NTCORE
    camToRobotR_ = rpyDegToMatrix(config::CAMERA_TO_ROBOT_ROLL_DEG,
                                  config::CAMERA_TO_ROBOT_PITCH_DEG,
                                  config::CAMERA_TO_ROBOT_YAW_DEG);
    camToRobotT_ = cv::Vec3d(config::CAMERA_TO_ROBOT_X,
                             config::CAMERA_TO_ROBOT_Y,
                             config::CAMERA_TO_ROBOT_Z);
    robotToCamR_ = camToRobotR_.t();
    robotToCamT_ = -matMul(robotToCamR_, camToRobotT_);
#endif
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "[NetworkPublisher] WSAStartup failed" << std::endl;
    }
#endif
}

NetworkPublisher::~NetworkPublisher() {
    stop();

    if (udpSocket_ != -1) {
#ifdef _WIN32
        closesocket(udpSocket_);
        WSACleanup();
#else
        close(udpSocket_);
#endif
    }

#ifdef USE_NTCORE
    if (ntConfigured_) {
        ntInstance_.StopClient();
    }
#endif
}

void NetworkPublisher::start() {
    if (running_) return;

    running_ = true;
    publishThread_ = std::thread(&NetworkPublisher::publishLoop, this);

    std::cout << "[NetworkPublisher] Started" << std::endl;
}

void NetworkPublisher::stop() {
    if (!running_) return;

    running_ = false;
    if (publishThread_.joinable()) {
        publishThread_.join();
    }

    std::cout << "[NetworkPublisher] Stopped" << std::endl;
}

void NetworkPublisher::publish(const VisionPayload& payload) {
    std::lock_guard<std::mutex> lock(queueMutex_);

    queue_.push_back(payload);

    // Keep queue size bounded
    while (queue_.size() > MAX_QUEUE_SIZE) {
        queue_.pop_front();
    }
}

bool NetworkPublisher::initUDP() {
    if (udpInitialized_) return true;

    udpSocket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (udpSocket_ < 0) {
        std::cerr << "[NetworkPublisher] Failed to create UDP socket" << std::endl;
        return false;
    }

    // Set non-blocking
#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(udpSocket_, FIONBIO, &mode);
#else
    int flags = fcntl(udpSocket_, F_GETFL, 0);
    fcntl(udpSocket_, F_SETFL, flags | O_NONBLOCK);
#endif

    std::memset(&udpAddr_, 0, sizeof(udpAddr_));
    udpAddr_.sin_family = AF_INET;
    udpAddr_.sin_port = htons(udpPort_);

    if (inet_pton(AF_INET, udpIp_.c_str(), &udpAddr_.sin_addr) <= 0) {
        std::cerr << "[NetworkPublisher] Invalid UDP address: " << udpIp_ << std::endl;
#ifdef _WIN32
        closesocket(udpSocket_);
#else
        close(udpSocket_);
#endif
        udpSocket_ = -1;
        return false;
    }

    udpInitialized_ = true;
    std::cout << "[NetworkPublisher] UDP initialized: " << udpIp_ << ":" << udpPort_ << std::endl;
    return true;
}

void NetworkPublisher::sendUDP(const std::string& json) {
    if (!udpInitialized_ && !initUDP()) {
        return;
    }

    sendto(udpSocket_, json.c_str(), json.size(), 0,
           (struct sockaddr*)&udpAddr_, sizeof(udpAddr_));
}

void NetworkPublisher::publishNetworkTables(const VisionPayload& payload) {
#ifdef USE_NTCORE
    if (!ntEnabled_) {
        return;
    }

    if (!ntConfigured_) {
        configureNetworkTables();
    }

    if (!visionTable_) {
        return;
    }

    timestampEntry_.SetDouble(payload.timestamp);
    latencyEntry_.SetDouble(payload.pipelineLatencyMs);
    fpsEntry_.SetDouble(payload.detectionRateHz);
    fastModeEntry_.SetBoolean(payload.fastModeActive);
    glareEntry_.SetBoolean(payload.glareSuppressed);

    std::vector<double> ids;
    std::vector<double> txVals;
    std::vector<double> tyVals;
    std::vector<double> taVals;
    std::vector<double> xyz;
    std::vector<double> rpy;
    std::vector<double> distances;
    std::vector<double> ambiguities;

    ids.reserve(payload.tags.size());
    txVals.reserve(payload.tags.size());
    tyVals.reserve(payload.tags.size());
    taVals.reserve(payload.tags.size());
    xyz.reserve(payload.tags.size() * 3);
    rpy.reserve(payload.tags.size() * 3);
    distances.reserve(payload.tags.size());
    ambiguities.reserve(payload.tags.size());

    const TagData* best = nullptr;
    double bestDist = std::numeric_limits<double>::max();
    if (payload.bestTagIndex >= 0 &&
        payload.bestTagIndex < static_cast<int>(payload.tags.size())) {
        best = &payload.tags[payload.bestTagIndex];
        bestDist = std::sqrt(best->tvec.dot(best->tvec));
    }

    auto pushEuler = [&](const cv::Vec3d& rvec) {
        cv::Matx33d R;
        cv::Mat Rmat;
        cv::Rodrigues(rvec, Rmat);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                R(r, c) = Rmat.at<double>(r, c);
            }
        }
        const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
        double roll;
        double yaw;
        if (std::abs(std::cos(pitch)) < 1e-6) {
            yaw = 0.0;
            roll = std::atan2(-R(0, 1), R(1, 1));
        } else {
            yaw = std::atan2(R(1, 0), R(0, 0));
            roll = std::atan2(R(2, 1), R(2, 2));
        }
        rpy.push_back(roll * 180.0 / M_PI);
        rpy.push_back(pitch * 180.0 / M_PI);
        rpy.push_back(yaw * 180.0 / M_PI);
    };

    for (const auto& tag : payload.tags) {
        ids.push_back(static_cast<double>(tag.id));
        txVals.push_back(tag.tx_deg);
        tyVals.push_back(tag.ty_deg);
        taVals.push_back(tag.ta_percent);
        xyz.push_back(tag.tvec[0]);
        xyz.push_back(tag.tvec[1]);
        xyz.push_back(tag.tvec[2]);
        pushEuler(tag.rvec);
        ambiguities.push_back(tag.poseAmbiguity);

        const double dist = std::sqrt(tag.tvec[0] * tag.tvec[0] +
                                      tag.tvec[1] * tag.tvec[1] +
                                      tag.tvec[2] * tag.tvec[2]);
        distances.push_back(dist);
        if (!best && dist < bestDist) {
            bestDist = dist;
            best = &tag;
        }
    }

    idsEntry_.SetDoubleArray(ids);
    txEntry_.SetDoubleArray(txVals);
    tyEntry_.SetDoubleArray(tyVals);
    taEntry_.SetDoubleArray(taVals);
    xyzEntry_.SetDoubleArray(xyz);
    rpyEntry_.SetDoubleArray(rpy);
    distanceEntry_.SetDoubleArray(distances);
    ambiguityEntry_.SetDoubleArray(ambiguities);

    if (best) {
        bestIdEntry_.SetDouble(static_cast<double>(best->id));
        bestDistanceEntry_.SetDouble(bestDist);
        bestPoseEntry_.SetDoubleArray({best->tvec[0], best->tvec[1], best->tvec[2]});

        cv::Mat Rmat;
        cv::Rodrigues(best->rvec, Rmat);
        double roll = std::atan2(Rmat.at<double>(2, 1), Rmat.at<double>(2, 2));
        double pitch = std::asin(std::clamp(-Rmat.at<double>(2, 0), -1.0, 1.0));
        double yaw = std::atan2(Rmat.at<double>(1, 0), Rmat.at<double>(0, 0));
        bestRpyEntry_.SetDoubleArray({
            roll * 180.0 / M_PI,
            pitch * 180.0 / M_PI,
            yaw * 180.0 / M_PI
        });
        bestTxPredEntry_.SetDouble(payload.bestTarget ? payload.bestTarget->predictedTxDeg : best->tx_deg);
        bestTyPredEntry_.SetDouble(payload.bestTarget ? payload.bestTarget->predictedTyDeg : best->ty_deg);
        bestClosingVelEntry_.SetDouble(payload.bestTarget ? payload.bestTarget->closingVelocityMps : best->closingVelocityMps);
        bestTimeToImpactEntry_.SetDouble(payload.bestTarget ? payload.bestTarget->timeToImpactMs : 0.0);
        bestStabilityEntry_.SetDouble(payload.bestTarget ? payload.bestTarget->stability : best->stabilityScore);
    } else {
        bestIdEntry_.SetDouble(-1.0);
        bestDistanceEntry_.SetDouble(0.0);
        bestPoseEntry_.SetDoubleArray({});
        bestRpyEntry_.SetDoubleArray({});
        bestTxPredEntry_.SetDouble(0.0);
        bestTyPredEntry_.SetDouble(0.0);
        bestClosingVelEntry_.SetDouble(0.0);
        bestTimeToImpactEntry_.SetDouble(0.0);
        bestStabilityEntry_.SetDouble(0.0);
    }

    connectedEntry_.SetBoolean(ntInstance_.IsConnected());

    if (payload.multiTag && payload.multiTag->valid) {
        multiTagCountEntry_.SetDouble(payload.multiTag->tagCount);
        multiTagAmbEntry_.SetDouble(payload.multiTag->avgAmbiguity);
    } else {
        multiTagCountEntry_.SetDouble(0.0);
        multiTagAmbEntry_.SetDouble(1.0);
    }

    if (limelightTable_) {
        llCameraPoseRobotEntry_.SetDoubleArray(poseVector(camToRobotT_, camToRobotR_));
        if (best) {
            llTvEntry_.SetDouble(1.0);
            llTidEntry_.SetDouble(static_cast<double>(best->id));
            llTsEntry_.SetDouble(best->skewDeg);
            llTlEntry_.SetDouble(payload.pipelineLatencyMs);
            llTshortEntry_.SetDouble(best->shortSidePx);
            llTlongEntry_.SetDouble(best->longSidePx);
            llThorEntry_.SetDouble(best->boundingWidthPx);
            llTvertEntry_.SetDouble(best->boundingHeightPx);
            llPoseAmbEntry_.SetDouble(best->poseAmbiguity);
            std::vector<double> tcx, tcy;
            for (const auto& corner : best->corners) {
                tcx.push_back(corner.x);
                tcy.push_back(corner.y);
            }
            llTcornXEntry_.SetDoubleArray(tcx);
            llTcornYEntry_.SetDoubleArray(tcy);
            llBestStabilityEntry_.SetDouble(payload.bestTarget ? payload.bestTarget->stability : best->stabilityScore);

            cv::Mat Rmat;
            cv::Rodrigues(best->rvec, Rmat);
            cv::Matx33d R_tc = matFromCv(Rmat);
            cv::Matx33d R_ct = R_tc.t();
            cv::Matx33d R_tag_to_robot = camToRobotR_ * R_tc;
            cv::Matx33d R_robot_to_tag = R_tag_to_robot.t();

            cv::Vec3d tagInRobot = matMul(camToRobotR_, best->tvec) + camToRobotT_;
            cv::Vec3d robotInTag = -matMul(R_robot_to_tag, tagInRobot);
            cv::Vec3d cameraInTag = -matMul(R_ct, best->tvec);

            llTargetPoseCamEntry_.SetDoubleArray(poseVector(best->tvec, R_tc));
            llTargetPoseRobotEntry_.SetDoubleArray(poseVector(tagInRobot, R_tag_to_robot));
            llCameraPoseTargetEntry_.SetDoubleArray(poseVector(cameraInTag, R_ct));
            llBotPoseTargetEntry_.SetDoubleArray(poseVector(robotInTag, R_robot_to_tag));
        } else {
            llTvEntry_.SetDouble(0.0);
            llTidEntry_.SetDouble(-1.0);
            llTsEntry_.SetDouble(0.0);
            llTlEntry_.SetDouble(payload.pipelineLatencyMs);
            llTshortEntry_.SetDouble(0.0);
            llTlongEntry_.SetDouble(0.0);
            llThorEntry_.SetDouble(0.0);
            llTvertEntry_.SetDouble(0.0);
            llPoseAmbEntry_.SetDouble(0.0);
            llTargetPoseCamEntry_.SetDoubleArray({});
            llTargetPoseRobotEntry_.SetDoubleArray({});
            llCameraPoseTargetEntry_.SetDoubleArray({});
            llBotPoseTargetEntry_.SetDoubleArray({});
            llTcornXEntry_.SetDoubleArray({});
            llTcornYEntry_.SetDoubleArray({});
            llBestStabilityEntry_.SetDouble(0.0);
        }

        llFpsEntry_.SetDouble(payload.detectionRateHz);
        llFastModeEntry_.SetBoolean(payload.fastModeActive);

        if (payload.multiTag && payload.multiTag->valid) {
            llBotPoseBlueEntry_.SetDoubleArray(poseVector(payload.multiTag->robotPoseField,
                                                         payload.multiTag->robotRotField));
            llCameraPoseFieldEntry_.SetDoubleArray(poseVector(payload.multiTag->cameraPoseField,
                                                             payload.multiTag->cameraRotField));
            auto redPose = mirrorBlueToRed(payload.multiTag->robotPoseField,
                                           payload.multiTag->robotRotField);
            llBotPoseRedEntry_.SetDoubleArray(poseVector(redPose.first, redPose.second));
            llBotPoseRobotEntry_.SetDoubleArray({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        } else {
            llBotPoseBlueEntry_.SetDoubleArray({});
            llBotPoseRedEntry_.SetDoubleArray({});
            llCameraPoseFieldEntry_.SetDoubleArray({});
            llBotPoseRobotEntry_.SetDoubleArray({});
        }
    }
#else
    (void)payload;
#endif
}

void NetworkPublisher::publishLoop() {
    while (running_) {
        VisionPayload payload;
        bool hasPayload = false;

        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (!queue_.empty()) {
                payload = queue_.front();
                queue_.pop_front();
                hasPayload = true;
            }
        }

        if (hasPayload) {
            // Build JSON payload
            std::ostringstream json;
            json << std::fixed << std::setprecision(4);
            json << "{";
            json << "\"timestamp\":" << payload.timestamp << ",";
            json << "\"pipeline_ms\":" << payload.pipelineLatencyMs << ",";
            json << "\"fps\":" << payload.detectionRateHz << ",";
            json << "\"fast_mode\":" << (payload.fastModeActive ? 1 : 0) << ",";
            json << "\"glare_active\":" << (payload.glareSuppressed ? 1 : 0) << ",";
            json << "\"tag_count\":" << payload.tags.size() << ",";
            json << "\"tagIDs\":\"";
            for (size_t i = 0; i < payload.tags.size(); i++) {
                if (i > 0) json << ",";
                json << payload.tags[i].id;
            }
            json << "\",";
            json << "\"tags\":[";

            for (size_t i = 0; i < payload.tags.size(); i++) {
                const auto& tag = payload.tags[i];
                if (i > 0) json << ",";
                json << "{";
                json << "\"id\":" << tag.id << ",";
                json << "\"tx\":" << tag.tx_deg << ",";
                json << "\"ty\":" << tag.ty_deg << ",";
                json << "\"ta\":" << tag.ta_percent << ",";
                json << "\"tvec\":[" << tag.tvec[0] << "," << tag.tvec[1] << "," << tag.tvec[2] << "],";
                json << "\"rvec\":[" << tag.rvec[0] << "," << tag.rvec[1] << "," << tag.rvec[2] << "],";
                json << "\"reproj_err\":" << tag.reprojError << ",";
                json << "\"ts\":" << tag.skewDeg << ",";
                json << "\"tshort\":" << tag.shortSidePx << ",";
                json << "\"tlong\":" << tag.longSidePx << ",";
                json << "\"thor\":" << tag.boundingWidthPx << ",";
                json << "\"tvert\":" << tag.boundingHeightPx << ",";
                json << "\"ambiguity\":" << tag.poseAmbiguity << ",";
                json << "\"decision_margin\":" << tag.decisionMargin << ",";
                json << "\"area_px\":" << tag.areaPx << ",";
                json << "\"distance_m\":" << tag.distanceM << ",";
                json << "\"closing_mps\":" << tag.closingVelocityMps << ",";
                json << "\"stability\":" << tag.stabilityScore << ",";
                json << "\"glare\":" << tag.glareFraction << ",";
                json << "\"predicted\":{";
                json << "\"tx\":" << tag.predictedTxDeg << ",";
                json << "\"ty\":" << tag.predictedTyDeg << ",";
                json << "\"time_ms\":" << tag.timeToImpactMs << "},";
                json << "\"corners\":[";
                for (size_t c = 0; c < tag.corners.size(); ++c) {
                    if (c > 0) json << ",";
                    json << "[" << tag.corners[c].x << "," << tag.corners[c].y << "]";
                }
                json << "]";
                json << "}";
            }
            json << "]";
            if (payload.bestTarget) {
                json << ",\"best\":{";
                json << "\"id\":" << payload.bestTarget->id << ",";
                json << "\"tx\":" << payload.bestTarget->tx_deg << ",";
                json << "\"ty\":" << payload.bestTarget->ty_deg << ",";
                json << "\"ta\":" << payload.bestTarget->ta_percent << ",";
                json << "\"distance_m\":" << payload.bestTarget->distanceM << ",";
                json << "\"stability\":" << payload.bestTarget->stability << ",";
                json << "\"predicted\":{";
                json << "\"tx\":" << payload.bestTarget->predictedTxDeg << ",";
                json << "\"ty\":" << payload.bestTarget->predictedTyDeg << "},";
                json << "\"closing_mps\":" << payload.bestTarget->closingVelocityMps << ",";
                json << "\"time_to_impact_ms\":" << payload.bestTarget->timeToImpactMs;
                json << "}";
            }
            if (payload.multiTag && payload.multiTag->valid) {
                json << ",\"multitag\":{";
                json << "\"count\":" << payload.multiTag->tagCount << ",";
                json << "\"avg_ambiguity\":" << payload.multiTag->avgAmbiguity << ",";
                auto botBlue = poseVector(payload.multiTag->robotPoseField, payload.multiTag->robotRotField);
                json << "\"botpose_wpiblue\":[";
                for (size_t b = 0; b < botBlue.size(); ++b) {
                    if (b > 0) json << ",";
                    json << botBlue[b];
                }
                json << "],";
                auto camField = poseVector(payload.multiTag->cameraPoseField, payload.multiTag->cameraRotField);
                json << "\"camerapose_fieldspace\":[";
                for (size_t b = 0; b < camField.size(); ++b) {
                    if (b > 0) json << ",";
                    json << camField[b];
                }
                json << "]";
                json << "}";
            }
            json << "}";

            std::string jsonStr = json.str();

            // Send via UDP if enabled
            if (udpEnabled_) {
                sendUDP(jsonStr);
            }

            // TODO: NetworkTables publishing
            publishNetworkTables(payload);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
}

#ifdef USE_NTCORE
void NetworkPublisher::configureNetworkTables() {
    if (ntConfigured_) {
        return;
    }

    ntInstance_ = nt::NetworkTableInstance::GetDefault();
    ntInstance_.StopClient();
    ntInstance_.StartClient4("AprilVision");
    ntInstance_.SetServer(ntServer_.c_str(), nt::NetworkTableInstance::kDefaultPort4);

    visionTable_ = ntInstance_.GetTable("vision");
    limelightTable_ = ntInstance_.GetTable("limelight");
    timestampEntry_ = visionTable_->GetEntry("timestamp");
    latencyEntry_ = visionTable_->GetEntry("pipeline_ms");
    idsEntry_ = visionTable_->GetEntry("ids");
    txEntry_ = visionTable_->GetEntry("tx");
    tyEntry_ = visionTable_->GetEntry("ty");
    taEntry_ = visionTable_->GetEntry("ta");
    xyzEntry_ = visionTable_->GetEntry("pose_xyz");
    rpyEntry_ = visionTable_->GetEntry("pose_rpy_deg");
    distanceEntry_ = visionTable_->GetEntry("distance_m");
    bestIdEntry_ = visionTable_->GetEntry("best/id");
    bestPoseEntry_ = visionTable_->GetEntry("best/xyz");
    bestRpyEntry_ = visionTable_->GetEntry("best/rpy_deg");
    bestDistanceEntry_ = visionTable_->GetEntry("best/distance_m");
    connectedEntry_ = visionTable_->GetEntry("connected");
    ambiguityEntry_ = visionTable_->GetEntry("pose_ambiguity");
    bestTxPredEntry_ = visionTable_->GetEntry("best/tx_pred_deg");
    bestTyPredEntry_ = visionTable_->GetEntry("best/ty_pred_deg");
    bestTimeToImpactEntry_ = visionTable_->GetEntry("best/time_to_impact_ms");
    bestClosingVelEntry_ = visionTable_->GetEntry("best/closing_mps");
    bestStabilityEntry_ = visionTable_->GetEntry("best/stability");
    multiTagCountEntry_ = visionTable_->GetEntry("multitag/count");
    multiTagAmbEntry_ = visionTable_->GetEntry("multitag/avg_ambiguity");
    fpsEntry_ = visionTable_->GetEntry("fps");
    fastModeEntry_ = visionTable_->GetEntry("fast_mode");
    glareEntry_ = visionTable_->GetEntry("glare_suppressed");

    if (limelightTable_) {
        llTvEntry_ = limelightTable_->GetEntry("tv");
        llTidEntry_ = limelightTable_->GetEntry("tid");
        llTsEntry_ = limelightTable_->GetEntry("ts");
        llTlEntry_ = limelightTable_->GetEntry("tl");
        llTshortEntry_ = limelightTable_->GetEntry("tshort");
        llTlongEntry_ = limelightTable_->GetEntry("tlong");
        llThorEntry_ = limelightTable_->GetEntry("thor");
        llTvertEntry_ = limelightTable_->GetEntry("tvert");
        llPoseAmbEntry_ = limelightTable_->GetEntry("poseambiguity");
        llTargetPoseCamEntry_ = limelightTable_->GetEntry("targetpose_cameraspace");
        llTargetPoseRobotEntry_ = limelightTable_->GetEntry("targetpose_robotspace");
        llCameraPoseRobotEntry_ = limelightTable_->GetEntry("camerapose_robotspace");
        llCameraPoseTargetEntry_ = limelightTable_->GetEntry("camerapose_targetspace");
        llBotPoseTargetEntry_ = limelightTable_->GetEntry("botpose_targetspace");
        llTcornXEntry_ = limelightTable_->GetEntry("tcornx");
        llTcornYEntry_ = limelightTable_->GetEntry("tcorny");
        llBotPoseBlueEntry_ = limelightTable_->GetEntry("botpose_wpiblue");
        llBotPoseRedEntry_ = limelightTable_->GetEntry("botpose_wpired");
        llBotPoseRobotEntry_ = limelightTable_->GetEntry("botpose_robotspace");
        llCameraPoseFieldEntry_ = limelightTable_->GetEntry("camerapose_fieldspace");
        llBestStabilityEntry_ = limelightTable_->GetEntry("target_stability");
        llFpsEntry_ = limelightTable_->GetEntry("pipelinefps");
        llFastModeEntry_ = limelightTable_->GetEntry("fast_mode");
    }
    ntConfigured_ = true;
    std::cout << "[NetworkPublisher] NetworkTables client started at "
              << ntServer_ << std::endl;
}
#endif