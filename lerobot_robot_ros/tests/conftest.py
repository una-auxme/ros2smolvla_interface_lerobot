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

"""Test fixtures that exercise the robot logic without a running ROS 2 graph."""

import pytest

from lerobot_robot_ros.config import ActionType, ROS2Config, ROS2InterfaceConfig
from lerobot_robot_ros.robot import ROS2Robot

ARM_JOINTS = [f"joint_{i}" for i in range(1, 7)]


class FakeInterface:
    """Stands in for ROS2Interface and records every command sent to it."""

    def __init__(self, config: ROS2InterfaceConfig):
        self.config = config
        self.is_connected = True
        self.joint_state = {
            "position": {f"{j}.pos": 0.0 for j in ARM_JOINTS},
            "velocity": {f"{j}.pos": 0.0 for j in ARM_JOINTS},
        }
        self.servo_calls = []
        self.joint_calls = []
        self.gripper_calls = []

    def servo(self, linear, angular):
        self.servo_calls.append((tuple(linear), tuple(angular)))

    def send_joint_position_command(self, joint_positions):
        self.joint_calls.append(list(joint_positions))

    def send_gripper_command(self, position):
        self.gripper_calls.append(position)
        return True


@pytest.fixture
def make_robot(tmp_path):
    def _make(action_type=ActionType.CARTESIAN_VELOCITY_TWIST_MSG, **overrides):
        interface_overrides = overrides.pop("ros2_interface", {})
        config = ROS2Config(
            action_type=action_type,
            calibration_dir=tmp_path,
            ros2_interface=ROS2InterfaceConfig(arm_joint_names=list(ARM_JOINTS), **interface_overrides),
            **overrides,
        )
        robot = ROS2Robot(config)
        robot.ros2_interface = FakeInterface(config.ros2_interface)
        return robot

    return _make


def velocity_action(vel=0.0, gripper=0.0):
    keys = ["linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"]
    return {f"{k}.vel": vel for k in keys} | {"gripper.pos": gripper}


def joint_action(pos=0.0, gripper=0.0):
    return {f"{j}.pos": pos for j in ARM_JOINTS} | {"gripper.pos": gripper}
