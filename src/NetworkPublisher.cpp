#include "NetworkPublisher.h"
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    std::vector<double> ids;
    std::vector<double> txVals;
    std::vector<double> tyVals;
    std::vector<double> taVals;
    std::vector<double> xyz;
    std::vector<double> rpy;
    std::vector<double> distances;

    ids.reserve(payload.tags.size());
    txVals.reserve(payload.tags.size());
    tyVals.reserve(payload.tags.size());
    taVals.reserve(payload.tags.size());
    xyz.reserve(payload.tags.size() * 3);
    rpy.reserve(payload.tags.size() * 3);
    distances.reserve(payload.tags.size());

    const TagData* best = nullptr;
    double bestDist = std::numeric_limits<double>::max();

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

        const double dist = std::sqrt(tag.tvec[0] * tag.tvec[0] +
                                      tag.tvec[1] * tag.tvec[1] +
                                      tag.tvec[2] * tag.tvec[2]);
        distances.push_back(dist);
        if (dist < bestDist) {
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
    } else {
        bestIdEntry_.SetDouble(-1.0);
        bestDistanceEntry_.SetDouble(0.0);
        bestPoseEntry_.SetDoubleArray({});
        bestRpyEntry_.SetDoubleArray({});
    }

    connectedEntry_.SetBoolean(ntInstance_.IsConnected());
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
                json << "\"reproj_err\":" << tag.reprojError;
                json << "}";
            }
            json << "]";
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

    ntConfigured_ = true;
    std::cout << "[NetworkPublisher] NetworkTables client started at "
              << ntServer_ << std::endl;
}
#endif