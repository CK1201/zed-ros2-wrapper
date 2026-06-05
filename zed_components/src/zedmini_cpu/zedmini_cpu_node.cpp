// Copyright 2026 Stereolabs
//
// CPU-only ZED Mini ROS 2 publisher backed by zed-open-capture.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <image_transport/camera_common.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/distortion_models.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "sensorcapture.hpp"
#include "videocapture.hpp"

namespace
{

constexpr char kLeftFrameId[] = "zed_left_camera_optical_frame";
constexpr char kRightFrameId[] = "zed_right_camera_optical_frame";
constexpr char kImuFrameId[] = "zed_imu_link";
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
constexpr uint64_t kLikelyUnixTimeNs = 1600000000000000000ULL;
constexpr int kMinWhiteBalanceParam = 28;
constexpr int kMaxWhiteBalanceParam = 65;

std::string trim(const std::string & input)
{
  const auto first = input.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last - first + 1);
}

std::string toLower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
  return value;
}

std::string shellQuote(const std::string & value)
{
  std::string quoted = "'";
  for (const char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

class CalibrationFile
{
public:
  bool load(const std::string & path)
  {
    std::ifstream input(path);
    if (!input.is_open()) {
      return false;
    }

    std::string section;
    std::string line;
    while (std::getline(input, line)) {
      auto comment_pos = line.find_first_of("#;");
      if (comment_pos != std::string::npos) {
        line = line.substr(0, comment_pos);
      }

      line = trim(line);
      if (line.empty()) {
        continue;
      }

      if (line.front() == '[' && line.back() == ']') {
        section = toLower(trim(line.substr(1, line.size() - 2)));
        continue;
      }

      const auto eq_pos = line.find('=');
      if (eq_pos == std::string::npos || section.empty()) {
        continue;
      }

      std::string key = toLower(trim(line.substr(0, eq_pos)));
      std::string value = trim(line.substr(eq_pos + 1));
      std::replace(value.begin(), value.end(), ',', '.');

      try {
        values_[section + ":" + key] = std::stod(value);
      } catch (const std::exception &) {
      }
    }

    return !values_.empty();
  }

  double get(const std::string & key, double default_value = 0.0) const
  {
    const auto found = values_.find(toLower(key));
    if (found == values_.end()) {
      return default_value;
    }
    return found->second;
  }

private:
  std::unordered_map<std::string, double> values_;
};

struct CameraCalibration
{
  cv::Mat map_left_x;
  cv::Mat map_left_y;
  cv::Mat map_right_x;
  cv::Mat map_right_y;
  sensor_msgs::msg::CameraInfo left_info;
  sensor_msgs::msg::CameraInfo right_info;
  std::string rectification_model;
  double baseline_m = 0.0;
};

std::string resolutionKey(int image_width)
{
  switch (image_width) {
    case 2208:
      return "2k";
    case 1920:
      return "fhd";
    case 1280:
      return "hd";
    case 672:
      return "vga";
    default:
      return "hd";
  }
}

double getStereoTranslation(
  const CalibrationFile & calibration,
  const std::string & axis,
  const std::string & resolution)
{
  const double fallback = calibration.get("stereo:" + axis, 0.0);
  return calibration.get("stereo:" + axis + "_" + resolution, fallback);
}

void copyMat3ToArray(const cv::Mat & matrix, std::array<double, 9> & output)
{
  output.fill(0.0);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      output[row * 3 + col] = matrix.at<double>(row, col);
    }
  }
}

void copyProjectionToArray(const cv::Mat & matrix, std::array<double, 12> & output)
{
  output.fill(0.0);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      output[row * 4 + col] = matrix.at<double>(row, col);
    }
  }
}

sensor_msgs::msg::CameraInfo makeCameraInfo(
  const cv::Size & image_size,
  const cv::Mat & rectification,
  const cv::Mat & projection,
  const cv::Mat & distortion,
  const std::string & distortion_model,
  const std::string & frame_id)
{
  sensor_msgs::msg::CameraInfo info;
  info.header.frame_id = frame_id;
  info.width = static_cast<uint32_t>(image_size.width);
  info.height = static_cast<uint32_t>(image_size.height);
  info.distortion_model = distortion_model;
  info.d.assign(static_cast<size_t>(distortion.total()), 0.0);
  for (int i = 0; i < distortion.rows * distortion.cols; ++i) {
    info.d[static_cast<size_t>(i)] = distortion.at<double>(i);
  }

  cv::Mat k = projection(cv::Rect(0, 0, 3, 3)).clone();
  copyMat3ToArray(k, info.k);
  copyMat3ToArray(rectification, info.r);
  copyProjectionToArray(projection, info.p);
  return info;
}

bool buildCalibration(
  const std::string & calibration_path,
  const cv::Size & image_size,
  CameraCalibration & output,
  std::string & error)
{
  CalibrationFile calibration;
  if (!calibration.load(calibration_path)) {
    error = "cannot load calibration file: " + calibration_path;
    return false;
  }

  const std::string res = resolutionKey(image_size.width);

  const auto readCameraMatrix = [&](const std::string & side) -> cv::Mat {
      const std::string prefix = side + "_cam_" + res + ":";
      const double fx = calibration.get(prefix + "fx", 0.0);
      const double fy = calibration.get(prefix + "fy", 0.0);
      const double cx = calibration.get(prefix + "cx", 0.0);
      const double cy = calibration.get(prefix + "cy", 0.0);
      cv::Mat matrix = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
      return matrix;
    };

  const auto readFisheyeDistCoeffs = [&](const std::string & side) -> cv::Mat {
      const std::string prefix = side + "_cam_" + res + ":";
      const double k1 = calibration.get(prefix + "k1", 0.0);
      const double k2 = calibration.get(prefix + "k2", 0.0);
      const double k3 = calibration.get(prefix + "k3", 0.0);
      const double k4 = calibration.get(prefix + "k4", 0.0);
      cv::Mat coeffs = (cv::Mat_<double>(4, 1) << k1, k2, k3, k4);
      return coeffs;
    };

  cv::Mat camera_left = readCameraMatrix("left");
  cv::Mat camera_right = readCameraMatrix("right");
  cv::Mat dist_left = readFisheyeDistCoeffs("left");
  cv::Mat dist_right = readFisheyeDistCoeffs("right");

  if (camera_left.at<double>(0, 0) <= 0.0 || camera_right.at<double>(0, 0) <= 0.0) {
    error = "calibration file does not contain valid intrinsics for resolution " + res;
    return false;
  }

  const double baseline = calibration.get("stereo:baseline", 0.0);
  const bool translations_are_mm = std::abs(baseline) > 1.0;
  const double translation_scale = translations_are_mm ? 0.001 : 1.0;
  const double tx = baseline * translation_scale;
  const double ty = getStereoTranslation(calibration, "ty", res) * translation_scale;
  const double tz = getStereoTranslation(calibration, "tz", res) * translation_scale;

  const cv::Mat rotation_vec = (cv::Mat_<double>(1, 3) <<
    getStereoTranslation(calibration, "rx", res),
    getStereoTranslation(calibration, "cv", res),
    getStereoTranslation(calibration, "rz", res));
  cv::Mat rotation;
  cv::Rodrigues(rotation_vec, rotation);

  cv::Mat r1;
  cv::Mat r2;
  cv::Mat p1;
  cv::Mat p2;
  cv::Mat q;
  cv::Mat translation = (cv::Mat_<double>(3, 1) << tx, ty, tz);
  cv::fisheye::stereoRectify(
    camera_left, dist_left, camera_right, dist_right, image_size, rotation, translation,
    r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY, image_size, 0.0, 1.0);

  cv::fisheye::initUndistortRectifyMap(
    camera_left, dist_left, r1, p1, image_size, CV_32FC1,
    output.map_left_x, output.map_left_y);
  cv::fisheye::initUndistortRectifyMap(
    camera_right, dist_right, r2, p2, image_size, CV_32FC1,
    output.map_right_x, output.map_right_y);

  output.left_info = makeCameraInfo(
    image_size, r1, p1, dist_left, sensor_msgs::distortion_models::EQUIDISTANT, kLeftFrameId);
  output.right_info = makeCameraInfo(
    image_size, r2, p2, dist_right, sensor_msgs::distortion_models::EQUIDISTANT, kRightFrameId);
  output.right_info.p[3] = -std::abs(output.left_info.p[0] * tx);
  output.rectification_model = "fisheye/equidistant";
  output.baseline_m = std::abs(tx);

  return true;
}

std::string defaultCalibrationPath(int serial_number)
{
  const char * home = std::getenv("HOME");
  if (home == nullptr || std::string(home).empty()) {
    throw std::runtime_error("HOME is not set; cannot locate ~/zed/settings");
  }

  std::filesystem::path path(home);
  path /= "zed";
  path /= "settings";
  path /= "SN" + std::to_string(serial_number) + ".conf";
  return path.string();
}

bool ensureCalibrationFile(int serial_number, std::string & calibration_path)
{
  calibration_path = defaultCalibrationPath(serial_number);
  if (std::filesystem::exists(calibration_path)) {
    return true;
  }

  std::filesystem::create_directories(std::filesystem::path(calibration_path).parent_path());

  const std::string url = "https://calib.stereolabs.com/?SN=" + std::to_string(serial_number);
  const std::string command =
    "wget -q " + shellQuote(url) + " -O " + shellQuote(calibration_path);
  const int ret = std::system(command.c_str());
  return ret == 0 && std::filesystem::exists(calibration_path) &&
         std::filesystem::file_size(calibration_path) > 0;
}

int serialFromSingleCalibrationFile()
{
  const char * home = std::getenv("HOME");
  if (home == nullptr || std::string(home).empty()) {
    return 0;
  }

  std::filesystem::path settings_dir(home);
  settings_dir /= "zed";
  settings_dir /= "settings";
  if (!std::filesystem::is_directory(settings_dir)) {
    return 0;
  }

  std::vector<int> serials;
  for (const auto & entry : std::filesystem::directory_iterator(settings_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    const std::string name = entry.path().filename().string();
    if (name.rfind("SN", 0) != 0 || entry.path().extension() != ".conf") {
      continue;
    }

    try {
      serials.push_back(std::stoi(name.substr(2, name.size() - 7)));
    } catch (const std::exception &) {
    }
  }

  if (serials.size() != 1) {
    return 0;
  }
  return serials.front();
}

sl_oc::video::RESOLUTION parseResolution(const std::string & value)
{
  const std::string normalized = toLower(value);
  if (normalized == "hd2k" || normalized == "2k") {
    return sl_oc::video::RESOLUTION::HD2K;
  }
  if (normalized == "hd1080" || normalized == "fhd") {
    return sl_oc::video::RESOLUTION::HD1080;
  }
  if (normalized == "hd720" || normalized == "hd" || normalized == "auto") {
    return sl_oc::video::RESOLUTION::HD720;
  }
  if (normalized == "vga") {
    return sl_oc::video::RESOLUTION::VGA;
  }
  return sl_oc::video::RESOLUTION::HD720;
}

sl_oc::video::FPS parseFps(int fps)
{
  if (fps <= 15) {
    return sl_oc::video::FPS::FPS_15;
  }
  if (fps <= 30) {
    return sl_oc::video::FPS::FPS_30;
  }
  if (fps <= 60) {
    return sl_oc::video::FPS::FPS_60;
  }
  return sl_oc::video::FPS::FPS_100;
}

int parseVideoDevice(const std::string & device, int fallback)
{
  const std::string trimmed = trim(device);
  const std::string normalized = toLower(trimmed);
  if (trimmed.empty() || normalized == "auto" || normalized == "-1") {
    return -1;
  }
  if (normalized == "camera_id") {
    return fallback;
  }

  const std::string prefix = "/dev/video";
  std::string number = trimmed;
  if (trimmed.rfind(prefix, 0) == 0) {
    number = trimmed.substr(prefix.size());
  }

  try {
    return std::stoi(number);
  } catch (const std::exception &) {
    return fallback;
  }
}

std::vector<int> findStereolabsVideoDevices()
{
  std::vector<int> devices;
  const std::filesystem::path video_root("/sys/class/video4linux");
  if (!std::filesystem::is_directory(video_root)) {
    return devices;
  }

  for (const auto & entry : std::filesystem::directory_iterator(video_root)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("video", 0) != 0) {
      continue;
    }

    int device_id = -1;
    try {
      device_id = std::stoi(name.substr(5));
    } catch (const std::exception &) {
      continue;
    }

    std::ifstream modalias_file(entry.path() / "device" / "modalias");
    std::string modalias;
    std::getline(modalias_file, modalias);
    modalias = toLower(modalias);
    if (modalias.find("usb:v2b03") != std::string::npos) {
      devices.push_back(device_id);
    }
  }

  std::sort(devices.begin(), devices.end());
  devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
  return devices;
}

std::string joinVideoDevices(const std::vector<int> & devices)
{
  std::ostringstream stream;
  for (size_t i = 0; i < devices.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << "/dev/video" << devices[i];
  }
  return stream.str();
}

int clampInt(int value, int low, int high)
{
  return std::max(low, std::min(value, high));
}

int parseYuvConversionCode(const std::string & format, std::string & normalized_format)
{
  std::string value = toLower(trim(format));
  std::replace(value.begin(), value.end(), '-', '_');

  if (value == "yuyv" || value == "yuy2" || value == "yuyv422") {
    normalized_format = "YUYV";
    return cv::COLOR_YUV2BGR_YUYV;
  }
  if (value == "yvyu") {
    normalized_format = "YVYU";
    return cv::COLOR_YUV2BGR_YVYU;
  }
  if (value == "uyvy" || value == "uyvy422") {
    normalized_format = "UYVY";
    return cv::COLOR_YUV2BGR_UYVY;
  }

  throw std::runtime_error(
    "unsupported video.yuv_format `" + format + "`; expected one of: YUYV, YVYU, UYVY");
}

sensor_msgs::msg::Image matToImage(
  const cv::Mat & image,
  const std_msgs::msg::Header & header)
{
  sensor_msgs::msg::Image msg;
  msg.header = header;
  msg.height = static_cast<uint32_t>(image.rows);
  msg.width = static_cast<uint32_t>(image.cols);
  msg.encoding = sensor_msgs::image_encodings::BGR8;
  msg.is_bigendian = false;
  msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(image.cols * image.elemSize());

  const size_t size = static_cast<size_t>(msg.step) * image.rows;
  msg.data.resize(size);
  if (image.isContinuous()) {
    std::copy(image.datastart, image.dataend, msg.data.begin());
  } else {
    for (int row = 0; row < image.rows; ++row) {
      const uint8_t * row_ptr = image.ptr<uint8_t>(row);
      std::copy(row_ptr, row_ptr + msg.step, msg.data.begin() + row * msg.step);
    }
  }
  return msg;
}

}  // namespace

class ZedMiniCpuNode : public rclcpp::Node
{
public:
  explicit ZedMiniCpuNode(const rclcpp::NodeOptions & options)
  : Node("zedmini_cpu_node", options)
  {
    initialize();
  }

  ~ZedMiniCpuNode() override
  {
    running_.store(false);
    if (video_thread_.joinable()) {
      video_thread_.join();
    }
    if (imu_thread_.joinable()) {
      imu_thread_.join();
    }
  }

private:
  template<typename T>
  T getParam(const std::string & name, const T & default_value)
  {
    if (!has_parameter(name)) {
      declare_parameter<T>(name, default_value);
    }

    T value = default_value;
    if (!get_parameter(name, value)) {
      return default_value;
    }
    return value;
  }

  void initialize()
  {
    const std::string camera_model = getParam<std::string>("general.camera_model", "zedm");
    if (camera_model != "zedm") {
      throw std::runtime_error("cpu_only mode currently supports only camera_model:=zedm");
    }

    use_pub_timestamps_ = getParam<bool>("debug.use_pub_timestamps", false);
    imu_pub_rate_ = getParam<double>("sensors.sensors_pub_rate", 100.0);
    yuv_conversion_code_ = parseYuvConversionCode(
      getParam<std::string>("video.yuv_format", "YUYV"), yuv_format_);

    sl_oc::video::VideoParams video_params;
    video_params.res = parseResolution(getParam<std::string>("general.grab_resolution", "HD720"));
    video_params.fps = parseFps(getParam<int>("general.grab_frame_rate", 30));
    video_params.verbose = sl_oc::VERBOSITY::WARNING;

    const int fallback_id = getParam<int>("general.camera_id", -1);
    const std::string video_device = getParam<std::string>("video_device", "auto");
    openVideo(video_params, video_device, fallback_id);

    const int serial_number = resolveSerialNumber();
    if (serial_number <= 0) {
      throw std::runtime_error(
        "cannot resolve ZED Mini serial number; pass serial_number:=<SN> or check HID access");
    }

    applyVideoControls();

    int full_width = 0;
    int height = 0;
    video_->getFrameSize(full_width, height);
    if (full_width <= 0 || height <= 0 || full_width % 2 != 0) {
      throw std::runtime_error("invalid ZED Mini side-by-side frame size");
    }
    image_size_ = cv::Size(full_width / 2, height);

    std::string calibration_path;
    if (!ensureCalibrationFile(serial_number, calibration_path)) {
      throw std::runtime_error("cannot download or locate ZED calibration file for serial " +
        std::to_string(serial_number));
    }

    std::string calibration_error;
    if (!buildCalibration(calibration_path, image_size_, calibration_, calibration_error)) {
      throw std::runtime_error(calibration_error);
    }

    sensors_ = std::make_unique<sl_oc::sensors::SensorCapture>(sl_oc::VERBOSITY::ERROR);
    if (!sensors_->initializeSensors(serial_number)) {
      throw std::runtime_error("cannot open ZED Mini IMU for serial " +
        std::to_string(serial_number));
    }
    video_->enableSensorSync(sensors_.get());

    createPublishers();

    RCLCPP_INFO(
      get_logger(),
      "ZED Mini CPU node ready: serial=%d size=%dx%d yuv_format=%s rectification=%s "
      "fx=%.3f cx=%.3f cy=%.3f baseline=%.5fm calibration=%s",
      serial_number, image_size_.width, image_size_.height, yuv_format_.c_str(),
      calibration_.rectification_model.c_str(),
      calibration_.left_info.p[0], calibration_.left_info.p[2], calibration_.left_info.p[6],
      calibration_.baseline_m,
      calibration_path.c_str());

    running_.store(true);
    video_thread_ = std::thread(&ZedMiniCpuNode::videoLoop, this);
    imu_thread_ = std::thread(&ZedMiniCpuNode::imuLoop, this);
  }

  void openVideo(
    const sl_oc::video::VideoParams & video_params,
    const std::string & video_device,
    int fallback_id)
  {
    const std::string normalized = toLower(trim(video_device));
    if (normalized == "auto" || normalized.empty() || normalized == "-1") {
      const auto candidates = findStereolabsVideoDevices();
      if (candidates.empty()) {
        throw std::runtime_error("cannot find any Stereolabs /dev/video* device");
      }

      for (const int candidate : candidates) {
        auto candidate_video = std::make_unique<sl_oc::video::VideoCapture>(video_params);
        RCLCPP_INFO(get_logger(), "Trying ZED video device /dev/video%d", candidate);
        if (candidate_video->initializeVideo(candidate)) {
          video_ = std::move(candidate_video);
          return;
        }
      }

      throw std::runtime_error(
        "cannot open ZED Mini video capture device; candidates tried: " +
        joinVideoDevices(candidates) + ". Stop other ZED processes or unplug/replug the camera.");
    }

    const int device_id = parseVideoDevice(video_device, fallback_id);
    video_ = std::make_unique<sl_oc::video::VideoCapture>(video_params);
    if (!video_->initializeVideo(device_id)) {
      throw std::runtime_error("cannot open ZED Mini video capture device: " + video_device);
    }
  }

  int resolveSerialNumber()
  {
    int serial_number = video_->getSerialNumber();
    if (serial_number > 0) {
      return serial_number;
    }

    const int configured_serial = getParam<int>("general.serial_number", 0);
    if (configured_serial > 0) {
      RCLCPP_WARN(
        get_logger(),
        "Could not read serial from video device; using serial_number parameter: %d",
        configured_serial);
      return configured_serial;
    }

    sl_oc::sensors::SensorCapture sensor_probe(sl_oc::VERBOSITY::ERROR);
    const auto sensor_serials = sensor_probe.getDeviceList(true);
    if (!sensor_serials.empty()) {
      if (sensor_serials.size() > 1) {
        RCLCPP_WARN(
          get_logger(),
          "Could not read serial from video device and multiple ZED sensor devices are present; using first serial: %d",
          sensor_serials.front());
      } else {
        RCLCPP_WARN(
          get_logger(),
          "Could not read serial from video device; using ZED sensor serial: %d",
          sensor_serials.front());
      }
      return sensor_serials.front();
    }

    serial_number = serialFromSingleCalibrationFile();
    if (serial_number > 0) {
      RCLCPP_WARN(
        get_logger(),
        "Could not read serial from video device or HID; using local calibration serial: %d",
        serial_number);
      return serial_number;
    }

    return 0;
  }

  void applyVideoControls()
  {
    video_->setBrightness(clampInt(getParam<int>("video.brightness", 4), 0, 8));
    video_->setContrast(clampInt(getParam<int>("video.contrast", 4), 0, 8));
    video_->setHue(clampInt(getParam<int>("video.hue", 0), 0, 11));
    video_->setSaturation(clampInt(getParam<int>("video.saturation", 4), 0, 8));
    video_->setSharpness(clampInt(getParam<int>("video.sharpness", 4), 0, 8));
    video_->setGamma(clampInt(getParam<int>("video.gamma", 8), 1, 9));

    const bool auto_exposure_gain = getParam<bool>("video.auto_exposure_gain", true);
    if (auto_exposure_gain) {
      const bool left_roi_reset = video_->resetROIforAECAGC(sl_oc::video::CAM_SENS_POS::LEFT);
      const bool right_roi_reset = video_->resetROIforAECAGC(sl_oc::video::CAM_SENS_POS::RIGHT);
      if (!left_roi_reset || !right_roi_reset) {
        RCLCPP_WARN(
          get_logger(),
          "Could not reset AEC/AGC ROI before enabling auto exposure/gain: left=%s right=%s",
          left_roi_reset ? "true" : "false",
          right_roi_reset ? "true" : "false");
      }
      video_->setAECAGC(true);
    } else {
      video_->setAECAGC(false);
      const int exposure = clampInt(getParam<int>("video.exposure", 80), 0, 100);
      const int gain = clampInt(getParam<int>("video.gain", 80), 0, 100);
      video_->setExposure(sl_oc::video::CAM_SENS_POS::LEFT, exposure);
      video_->setExposure(sl_oc::video::CAM_SENS_POS::RIGHT, exposure);
      video_->setGain(sl_oc::video::CAM_SENS_POS::LEFT, gain);
      video_->setGain(sl_oc::video::CAM_SENS_POS::RIGHT, gain);
    }

    const bool auto_white_balance = getParam<bool>("video.auto_whitebalance", true);
    const int white_balance_param = clampInt(
      getParam<int>("video.whitebalance_temperature", 42),
      kMinWhiteBalanceParam,
      kMaxWhiteBalanceParam);
    const int white_balance = (
      kMinWhiteBalanceParam + kMaxWhiteBalanceParam - white_balance_param) * 100;
    video_->setAutoWhiteBalance(auto_white_balance);
    if (!auto_white_balance) {
      video_->setWhiteBalance(white_balance);
    }

    RCLCPP_INFO(
      get_logger(),
      "Video controls applied: brightness=%d contrast=%d hue=%d saturation=%d sharpness=%d gamma=%d "
      "auto_exposure_gain=%s exposure_l=%d exposure_r=%d gain_l=%d gain_r=%d "
      "auto_whitebalance=%s whitebalance_param=%d whitebalance=%d",
      video_->getBrightness(), video_->getContrast(), video_->getHue(),
      video_->getSaturation(), video_->getSharpness(), video_->getGamma(),
      video_->getAECAGC() ? "true" : "false",
      video_->getExposure(sl_oc::video::CAM_SENS_POS::LEFT),
      video_->getExposure(sl_oc::video::CAM_SENS_POS::RIGHT),
      video_->getGain(sl_oc::video::CAM_SENS_POS::LEFT),
      video_->getGain(sl_oc::video::CAM_SENS_POS::RIGHT),
      video_->getAutoWhiteBalance() ? "true" : "false",
      white_balance_param,
      video_->getWhiteBalance());
  }

  void createPublishers()
  {
    rclcpp::QoS image_qos(5);
    image_qos.reliable();
    const auto image_rmw_qos = image_qos.get_rmw_qos_profile();
    const auto sensor_qos = rclcpp::SensorDataQoS();

    setImageTransportPlugins("~/left/color/rect/image");
    setImageTransportPlugins("~/right/color/rect/image");

    left_image_pub_ = image_transport::create_publisher(
      this, "~/left/color/rect/image", image_rmw_qos);
    right_image_pub_ = image_transport::create_publisher(
      this, "~/right/color/rect/image", image_rmw_qos);

    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      "~/left/color/rect/camera_info", image_qos);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      "~/right/color/rect/camera_info", image_qos);
    left_info_transport_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      "~/left/color/rect/image/camera_info", image_qos);
    right_info_transport_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      "~/right/color/rect/image/camera_info", image_qos);
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("~/imu/data", sensor_qos);

    RCLCPP_INFO(get_logger(), " * Advertised on topic: %s", left_image_pub_.getTopic().c_str());
    RCLCPP_INFO(get_logger(), " * Advertised on topic: %s", right_image_pub_.getTopic().c_str());
    RCLCPP_INFO(get_logger(), " * Advertised on topic: %s", left_info_pub_->get_topic_name());
    RCLCPP_INFO(get_logger(), " * Advertised on topic: %s", right_info_pub_->get_topic_name());
    RCLCPP_INFO(get_logger(), " * Advertised on topic: %s",
      left_info_transport_pub_->get_topic_name());
    RCLCPP_INFO(get_logger(), " * Advertised on topic: %s",
      right_info_transport_pub_->get_topic_name());
    RCLCPP_INFO(get_logger(), " * Advertised on topic: %s", imu_pub_->get_topic_name());
  }

  void setImageTransportPlugins(const std::string & topic)
  {
    std::string resolved = rclcpp::expand_topic_or_service_name(
      topic, get_name(), get_namespace());
    const auto namespace_len = get_effective_namespace().length();
    std::string param_base = resolved.substr(namespace_len);
    std::replace(param_base.begin(), param_base.end(), '/', '.');
    if (!param_base.empty() && param_base.front() == '.') {
      param_base = param_base.substr(1);
    }

    std::vector<std::string> allowed;
    try {
      const auto declared = image_transport::getDeclaredTransports();
      for (const auto & transport : declared) {
        if (transport.find("/compressedDepth") == std::string::npos) {
          allowed.push_back(transport);
        }
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        get_logger(), "Could not inspect image_transport plugins for %s: %s",
        topic.c_str(), e.what());
      return;
    }

    if (allowed.empty()) {
      return;
    }

    try {
      declare_parameter(param_base + ".enable_pub_plugins", allowed);
    } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
      RCLCPP_WARN(
        get_logger(),
        "Parameter %s.enable_pub_plugins is already declared; using existing image_transport plugin list",
        param_base.c_str());
    }
  }

  rclcpp::Time stampFromCameraTime(uint64_t timestamp_ns)
  {
    if (use_pub_timestamps_ || timestamp_ns < kLikelyUnixTimeNs) {
      return now();
    }
    return rclcpp::Time(timestamp_ns, RCL_SYSTEM_TIME);
  }

  void publishCameraInfos(const rclcpp::Time & stamp)
  {
    auto left_info = calibration_.left_info;
    auto right_info = calibration_.right_info;
    left_info.header.stamp = stamp;
    right_info.header.stamp = stamp;

    left_info_pub_->publish(left_info);
    right_info_pub_->publish(right_info);
    left_info_transport_pub_->publish(left_info);
    right_info_transport_pub_->publish(right_info);
  }

  void videoLoop()
  {
    uint64_t last_timestamp = 0;
    uint64_t last_frame_id = 0;
    uint64_t null_frames = 0;
    uint64_t empty_initial_frames = 0;
    uint64_t repeated_timestamps = 0;
    uint64_t repeated_frame_ids = 0;
    uint64_t published_frames = 0;
    auto last_wait_log = std::chrono::steady_clock::now();
    auto first_wait = last_wait_log;
    cv::Mat frame_bgr;
    cv::Mat left_rect;
    cv::Mat right_rect;

    while (rclcpp::ok() && running_.load()) {
      const auto & frame = video_->getLastFrame(100);
      if (!running_.load()) {
        break;
      }
      const bool empty_initial_frame = frame.data != nullptr && frame.timestamp == 0 &&
        frame.frame_id == 0;
      if (frame.data == nullptr) {
        ++null_frames;
      } else if (empty_initial_frame) {
        ++empty_initial_frames;
      } else if (frame.timestamp != 0 && frame.timestamp == last_timestamp) {
        ++repeated_timestamps;
      } else if (frame.timestamp == 0 && frame.frame_id != 0 && frame.frame_id == last_frame_id) {
        ++repeated_frame_ids;
      }

      if (frame.data == nullptr || empty_initial_frame ||
        (frame.timestamp != 0 && frame.timestamp == last_timestamp) ||
        (frame.timestamp == 0 && frame.frame_id != 0 && frame.frame_id == last_frame_id))
      {
        const auto now_time = std::chrono::steady_clock::now();
        if (published_frames == 0 &&
          now_time - last_wait_log > std::chrono::seconds(1))
        {
          const auto waited_sec =
            std::chrono::duration_cast<std::chrono::seconds>(now_time - first_wait).count();
          if (waited_sec >= 5 && empty_initial_frames > 0 &&
            null_frames == 0 && repeated_timestamps == 0 && repeated_frame_ids == 0)
          {
            RCLCPP_WARN(
              get_logger(),
              "Waiting for real video frames: the UVC driver is returning only empty initial frames "
              "(frame_id=0 timestamp=0) for %lds. Check USB3 cable/port, camera power, and kernel "
              "uvcvideo errors.",
              static_cast<long>(waited_sec));
          } else {
            RCLCPP_WARN(
              get_logger(),
              "Waiting for video frames: null=%lu empty_initial=%lu repeated_ts=%lu repeated_frame_id=%lu last_ts=%lu last_frame_id=%lu",
              null_frames, empty_initial_frames, repeated_timestamps, repeated_frame_ids,
              last_timestamp, last_frame_id);
          }
          last_wait_log = now_time;
        }
        continue;
      }
      last_timestamp = frame.timestamp;
      last_frame_id = frame.frame_id;

      try {
        cv::Mat frame_yuv(frame.height, frame.width, CV_8UC2, frame.data);
        cv::cvtColor(frame_yuv, frame_bgr, yuv_conversion_code_);

        cv::Mat left_raw = frame_bgr(cv::Rect(0, 0, frame_bgr.cols / 2, frame_bgr.rows));
        cv::Mat right_raw = frame_bgr(cv::Rect(frame_bgr.cols / 2, 0, frame_bgr.cols / 2,
          frame_bgr.rows));

        cv::remap(
          left_raw, left_rect, calibration_.map_left_x, calibration_.map_left_y,
          cv::INTER_LINEAR);
        cv::remap(
          right_raw, right_rect, calibration_.map_right_x, calibration_.map_right_y,
          cv::INTER_LINEAR);

        const auto stamp = stampFromCameraTime(frame.timestamp);
        std_msgs::msg::Header left_header;
        left_header.stamp = stamp;
        left_header.frame_id = kLeftFrameId;
        std_msgs::msg::Header right_header;
        right_header.stamp = stamp;
        right_header.frame_id = kRightFrameId;

        auto left_msg = matToImage(left_rect, left_header);
        auto right_msg = matToImage(right_rect, right_header);
        left_image_pub_.publish(left_msg);
        right_image_pub_.publish(right_msg);
        publishCameraInfos(stamp);
        if (published_frames == 0) {
          RCLCPP_INFO(
            get_logger(),
            "First video frame published: frame=%ux%u left=%dx%d frame_id=%lu timestamp=%lu",
            frame.width, frame.height, left_rect.cols, left_rect.rows, frame.frame_id,
            frame.timestamp);
        }
        ++published_frames;
      } catch (const std::exception & e) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Video publishing failed: %s", e.what());
      }
    }
  }

  void imuLoop()
  {
    uint64_t last_publish_timestamp = 0;
    uint64_t min_period_ns = 0;
    if (imu_pub_rate_ > 0.0) {
      min_period_ns = static_cast<uint64_t>(1e9 / imu_pub_rate_);
    }

    while (rclcpp::ok() && running_.load()) {
      const auto imu_data = sensors_->getLastIMUData(5000);
      if (!running_.load()) {
        break;
      }
      if (imu_data.valid != sl_oc::sensors::data::Imu::NEW_VAL || imu_data.timestamp == 0) {
        continue;
      }
      if (min_period_ns > 0 && last_publish_timestamp > 0 &&
        imu_data.timestamp - last_publish_timestamp < min_period_ns)
      {
        continue;
      }
      last_publish_timestamp = imu_data.timestamp;

      sensor_msgs::msg::Imu msg;
      msg.header.stamp = stampFromCameraTime(imu_data.timestamp);
      msg.header.frame_id = kImuFrameId;

      msg.orientation_covariance[0] = -1.0;
      msg.angular_velocity.x = static_cast<double>(imu_data.gX) * kDegToRad;
      msg.angular_velocity.y = static_cast<double>(imu_data.gY) * kDegToRad;
      msg.angular_velocity.z = static_cast<double>(imu_data.gZ) * kDegToRad;
      msg.linear_acceleration.x = imu_data.aX;
      msg.linear_acceleration.y = imu_data.aY;
      msg.linear_acceleration.z = imu_data.aZ;
      imu_pub_->publish(msg);
    }
  }

  std::unique_ptr<sl_oc::video::VideoCapture> video_;
  std::unique_ptr<sl_oc::sensors::SensorCapture> sensors_;
  CameraCalibration calibration_;
  cv::Size image_size_;

  image_transport::Publisher left_image_pub_;
  image_transport::Publisher right_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_transport_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_transport_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;

  std::atomic_bool running_{false};
  std::thread video_thread_;
  std::thread imu_thread_;
  bool use_pub_timestamps_{false};
  double imu_pub_rate_{100.0};
  std::string yuv_format_{"YUYV"};
  int yuv_conversion_code_{cv::COLOR_YUV2BGR_YUYV};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto options = rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true);
    auto node = std::make_shared<ZedMiniCpuNode>(options);
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("zedmini_cpu_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
