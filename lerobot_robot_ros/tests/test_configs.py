# Copyright 2025-2026 University of Augsburg, Chair of Mechatronics by Noah Böckmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checks over every robot config this package registers."""

import pytest
from lerobot.robots import RobotConfig
from lerobot.robots.utils import make_robot_from_config

import lerobot_robot_ros  # noqa: F401  (registers the configs)

CONFIG_TYPES = sorted(
    name
    for name in RobotConfig.get_known_choices()
    if RobotConfig.get_choice_class(name).__module__.startswith("lerobot_robot_ros")
)


def test_configs_registered():
    assert "ur_10e_real" in CONFIG_TYPES
    assert "ur_10e_sim" in CONFIG_TYPES


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
def test_config_builds_its_robot(config_type, tmp_path):
    config = RobotConfig.get_choice_class(config_type)(calibration_dir=tmp_path)
    assert config.type == config_type
    robot = make_robot_from_config(config)
    # lerobot resolves "<Name>Config" to the robot class "<Name>"
    assert type(robot).__name__ + "Config" == type(config).__name__
    assert robot.observation_features
    assert robot.action_features


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
def test_camera_crop_and_resize_match_declared_shape(config_type):
    """The declared (height, width) is the dataset feature shape; crop/resize must produce exactly that."""
    for name, camera in RobotConfig.get_choice_class(config_type)().cameras.items():
        resize = getattr(camera, "resize", None)  # (width, height), applied last
        crop_box = getattr(camera, "crop_box", None)  # (ymin, ymax, xmin, xmax)
        if resize is not None:
            produced = (resize[1], resize[0])
        elif crop_box is not None:
            produced = (crop_box[1] - crop_box[0], crop_box[3] - crop_box[2])
        else:
            continue
        assert produced == (camera.height, camera.width), f"{config_type}.{name}"
