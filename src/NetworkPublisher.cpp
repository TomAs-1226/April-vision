#include "NetworkPublisher.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#ifndef _WIN32
#include <fcntl.h>
#endif

#ifdef APRILV_HAS_NTCORE
#include <networktables/NetworkTableInstance.h>
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

void NetworkPublisher::setNtServer(const std::string& server) {
    std::lock_guard<std::mutex> lock(ntMutex_);
    if (server.empty() || ntServer_ == server) return;
    ntServer_ = server;
#ifdef APRILV_HAS_NTCORE
    ntReady_ = false;
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

#ifdef APRILV_HAS_NTCORE
    if (connectionListener_) {
        ntInstance_.RemoveConnectionListener(*connectionListener_);
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

            if (ntEnabled_) {
                publishNetworkTables(payload);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
}

void NetworkPublisher::ensureNetworkTables() {
#ifdef APRILV_HAS_NTCORE
    std::string serverCopy;
    {
        std::lock_guard<std::mutex> lock(ntMutex_);
        if (ntReady_) return;
        serverCopy = ntServer_;
    }

    if (connectionListener_) {
        ntInstance_.RemoveConnectionListener(*connectionListener_);
        connectionListener_.reset();
    }
    ntInstance_.StopClient();
    ntInstance_ = nt::NetworkTableInstance::Create();
    ntInstance_.StartClient(serverCopy.c_str(), 5810);
    table_ = ntInstance_.GetTable("vision");
    connectionListener_ = ntInstance_.AddConnectionListener(
        [this](const nt::Event& event) {
            if (event.Is(nt::EventFlags::kConnected)) {
                std::cout << "[NetworkTables] Connected to " << ntServer_ << std::endl;
            } else if (event.Is(nt::EventFlags::kDisconnected)) {
                std::cout << "[NetworkTables] Disconnected from server" << std::endl;
            }
        });
    std::lock_guard<std::mutex> lock(ntMutex_);
    ntReady_ = true;
#else
    static bool warned = false;
    if (!warned) {
        std::cout << "[NetworkPublisher] NetworkTables support not built; UDP only" << std::endl;
        warned = true;
    }
#endif
}

void NetworkPublisher::publishNetworkTables(const VisionPayload& payload) {
#ifdef APRILV_HAS_NTCORE
    ensureNetworkTables();
    {
        std::lock_guard<std::mutex> lock(ntMutex_);
        if (!ntReady_ || !table_) return;
    }

    table_->GetEntry("timestamp").SetDouble(payload.timestamp);
    table_->GetEntry("tag_count").SetDouble(static_cast<double>(payload.tags.size()));

    std::vector<double> ids, txs, tys, tas, reproj;
    std::vector<double> tvecs, rvecs;
    ids.reserve(payload.tags.size());
    txs.reserve(payload.tags.size());
    tys.reserve(payload.tags.size());
    tas.reserve(payload.tags.size());
    reproj.reserve(payload.tags.size());
    tvecs.reserve(payload.tags.size() * 3);
    rvecs.reserve(payload.tags.size() * 3);

    for (const auto& tag : payload.tags) {
        ids.push_back(tag.id);
        txs.push_back(tag.tx_deg);
        tys.push_back(tag.ty_deg);
        tas.push_back(tag.ta_percent);
        reproj.push_back(tag.reprojError);
        tvecs.push_back(tag.tvec[0]);
        tvecs.push_back(tag.tvec[1]);
        tvecs.push_back(tag.tvec[2]);
        rvecs.push_back(tag.rvec[0]);
        rvecs.push_back(tag.rvec[1]);
        rvecs.push_back(tag.rvec[2]);
    }

    table_->GetEntry("ids").SetDoubleArray(ids);
    table_->GetEntry("tx_deg").SetDoubleArray(txs);
    table_->GetEntry("ty_deg").SetDoubleArray(tys);
    table_->GetEntry("ta_percent").SetDoubleArray(tas);
    table_->GetEntry("reproj_error").SetDoubleArray(reproj);
    table_->GetEntry("tvec").SetDoubleArray(tvecs);
    table_->GetEntry("rvec").SetDoubleArray(rvecs);
#else
    (void)payload;
    ensureNetworkTables();
#endif
}