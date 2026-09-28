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

import numpy as np
import pytest
from conftest import POSE_KEYS, FakeInterface

from lerobot_robot_ros.config import CartesianRobotSimNoMiddleConfig
from lerobot_robot_ros.robot import CartesianRobot, rotate_pose9_z

POLICY_POSE = [0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
ROBOT_POSE = [-0.4, -0.1, 0.3, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]  # POLICY_POSE rotated by 180 deg about z


@pytest.fixture
def nora(tmp_path):
    robot = CartesianRobot(CartesianRobotSimNoMiddleConfig(cameras={}, calibration_dir=tmp_path))
    robot.ros2_interface = FakeInterface(robot.config.ros2_interface)
    robot.ros2_interface.joint_state["cart_pose"] = dict(zip(POSE_KEYS, ROBOT_POSE, strict=True))
    return robot


def test_rotate_pose9_z_roundtrip():
    pose = np.arange(9.0)
    np.testing.assert_allclose(rotate_pose9_z(rotate_pose9_z(pose, 37.0), 37.0, inverse=True), pose, atol=1e-12)
    np.testing.assert_allclose(rotate_pose9_z(ROBOT_POSE, 180.0), POLICY_POSE, atol=1e-12)


def test_action_features(nora):
    assert list(nora.action_features) == [f"target_robot_pose.{n}.pos" for n in range(9)]


def test_observation_in_policy_frame(nora):
    obs = nora.get_observation()
    assert list(obs) == [f"current_robot_pose.{n}.pos" for n in range(9)]
    np.testing.assert_allclose(list(obs.values()), POLICY_POSE, atol=1e-12)


def test_action_sent_in_robot_frame(nora):
    action = {f"target_robot_pose.{n}.pos": v for n, v in enumerate(POLICY_POSE)}
    sent = nora.send_action(action)
    np.testing.assert_allclose(nora.ros2_interface.pose_calls[-1], ROBOT_POSE, atol=1e-12)
    assert list(sent) == list(action)
    np.testing.assert_allclose(list(sent.values()), POLICY_POSE, atol=1e-12)


def test_reports_limited_pose_in_policy_frame(nora):
    # Interface limit halves the robot-frame x: -0.4 -> -0.2, i.e. +0.2 in the policy frame
    nora.ros2_interface.pose_result = lambda pose: np.asarray(pose) * [0.5, 1, 1, 1, 1, 1, 1, 1, 1]
    sent = nora.send_action({f"target_robot_pose.{n}.pos": v for n, v in enumerate(POLICY_POSE)})
    assert sent["target_robot_pose.0.pos"] == pytest.approx(0.2)


def test_observation_replay_returns_to_start(nora):
    nora.send_action(nora.get_observation())
    np.testing.assert_allclose(nora.ros2_interface.pose_calls[-1], ROBOT_POSE, atol=1e-12)


def test_non_finite_pose_dropped(nora):
    action = {f"target_robot_pose.{n}.pos": v for n, v in enumerate(POLICY_POSE)}
    action["target_robot_pose.4.pos"] = float("nan")
    nora.send_action(action)
    assert nora.ros2_interface.pose_calls == []
