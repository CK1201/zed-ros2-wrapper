#!/usr/bin/env python3
#
# Copyright 2026 Stereolabs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import unittest


class ZedMiniCpuRectificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).parents[1] / "src" / "zedmini_cpu" / "zedmini_cpu_node.cpp"
        cls.code = source.read_text(encoding="utf-8")

    def test_zedmini_cpu_uses_fisheye_rectification(self):
        code = self.code

        self.assertIn("cv::fisheye::stereoRectify", code)
        self.assertIn("cv::fisheye::initUndistortRectifyMap", code)
        self.assertIn("sensor_msgs::distortion_models::EQUIDISTANT", code)
        self.assertIn("cv::Mat_<double>(4, 1)", code)
        self.assertIn('prefix + "k4"', code)
        self.assertIn("output.right_info.p[3] = -std::abs", code)
        self.assertNotIn('distortion_model = "plumb_bob"', code)

    def test_zedmini_cpu_uses_timer_for_imu_publishing(self):
        code = self.code

        self.assertIn("create_wall_timer", code)
        self.assertIn("rclcpp::TimerBase::SharedPtr imu_timer_", code)
        self.assertIn("last_imu_publish_timestamp_", code)
        self.assertIn("imu_data.timestamp == last_imu_publish_timestamp_", code)
        self.assertNotIn("std::thread imu_thread_", code)
        self.assertNotIn("imu_data.timestamp - last_publish_timestamp < min_period_ns", code)

    def test_zedmini_cpu_uses_timer_for_image_publishing(self):
        code = self.code

        self.assertIn("rclcpp::TimerBase::SharedPtr video_timer_", code)
        self.assertIn("void publishLatestVideoFrame()", code)
        self.assertIn("video_publish_in_progress_", code)
        self.assertIn("frame.timestamp == last_video_timestamp_", code)
        self.assertIn("frame.frame_id == last_video_frame_id_", code)
        self.assertNotIn("std::thread video_thread_", code)
        self.assertNotIn("void videoLoop()", code)

    def test_zedmini_cpu_has_no_imu_while_loop(self):
        code = self.code

        self.assertNotIn("void imuLoop()", code)
        self.assertIn("void publishLatestImu()", code)

    def test_zedmini_cpu_reports_video_diagnostics(self):
        code = self.code

        self.assertIn("debug.video_diagnostics_period_sec", code)
        self.assertIn('getParam<double>("debug.video_diagnostics_period_sec", 0.0)', code)
        self.assertIn("video_diagnostics_enabled_", code)
        self.assertIn("Video diagnostics:", code)
        self.assertIn("video_timer_ticks_", code)
        self.assertIn("video_stats_published_frames_", code)
        self.assertIn("video_busy_skips_", code)
        self.assertIn("video_null_frame_skips_", code)
        self.assertIn("video_repeated_timestamp_skips_", code)
        self.assertIn("video_repeated_frame_id_skips_", code)
        self.assertIn("video_timer_late_ticks_", code)
        self.assertIn("video_processing_overruns_", code)
        self.assertIn("video_processing_total_ns_", code)
        self.assertIn("video_processing_max_ns_", code)

    def test_zedmini_cpu_releases_video_before_sensors(self):
        code = self.code

        self.assertIn("shutdownCapture();", code)
        self.assertIn("void shutdownCapture()", code)
        self.assertLess(code.index("video_.reset();"), code.index("sensors_.reset();"))

    def test_zedmini_cpu_fails_when_video_serial_is_unreadable(self):
        code = self.code

        self.assertIn("Video serial is not readable after opening", code)
        self.assertIn("replug the camera", code)


if __name__ == "__main__":
    unittest.main()
