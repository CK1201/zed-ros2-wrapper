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
    def test_zedmini_cpu_uses_fisheye_rectification(self):
        source = Path(__file__).parents[1] / "src" / "zedmini_cpu" / "zedmini_cpu_node.cpp"
        code = source.read_text(encoding="utf-8")

        self.assertIn("cv::fisheye::stereoRectify", code)
        self.assertIn("cv::fisheye::initUndistortRectifyMap", code)
        self.assertIn("sensor_msgs::distortion_models::EQUIDISTANT", code)
        self.assertIn("cv::Mat_<double>(4, 1)", code)
        self.assertIn('prefix + "k4"', code)
        self.assertIn("output.right_info.p[3] = -std::abs", code)
        self.assertNotIn('distortion_model = "plumb_bob"', code)


if __name__ == "__main__":
    unittest.main()
