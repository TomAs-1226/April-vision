#include "NetworkPublisher.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#ifndef _WIN32
#include <fcntl.h>
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
            // This requires linking against ntcore library
            // For now, UDP fallback is implemented
            if (ntEnabled_) {
                // Would publish to NetworkTables here
                // nt::NetworkTableInstance::GetDefault()...
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
}