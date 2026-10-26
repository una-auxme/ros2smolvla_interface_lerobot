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
from conftest import POSE_KEYS, POSE_QUAT_KEYS, FakeInterface
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ros.config import CartesianRobotSimNoMiddleConfig
from lerobot_robot_ros.robot import CartesianRobot, rotate_pose_z

DATA_FRAME_POSE = [0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
ROBOT_POSE = [-0.4, -0.1, 0.3, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]  # DATA_FRAME_POSE rotated by 180 deg about z


@pytest.fixture
def robot(tmp_path):
    robot = CartesianRobot(CartesianRobotSimNoMiddleConfig(cameras={}, calibration_dir=tmp_path))
    robot.ros2_interface = FakeInterface(robot.config.ros2_interface)
    robot.ros2_interface.joint_state["cart_pose"] = dict(zip(POSE_KEYS, ROBOT_POSE, strict=True))
    return robot


def test_rotate_pose_z_roundtrip():
    pose = np.arange(9.0)
    np.testing.assert_allclose(rotate_pose_z(rotate_pose_z(pose, 37.0), 37.0, inverse=True), pose, atol=1e-12)
    np.testing.assert_allclose(rotate_pose_z(ROBOT_POSE, 180.0), DATA_FRAME_POSE, atol=1e-12)


def test_action_features(robot):
    assert list(robot.action_features) == [f"target_robot_pose.{n}.pos" for n in range(9)]


def test_observation_in_data_frame(robot):
    obs = robot.get_observation()
    assert list(obs) == [f"current_robot_pose.{n}.pos" for n in range(9)]
    np.testing.assert_allclose(list(obs.values()), DATA_FRAME_POSE, atol=1e-12)


def test_action_sent_in_robot_frame(robot):
    action = {f"target_robot_pose.{n}.pos": v for n, v in enumerate(DATA_FRAME_POSE)}
    sent = robot.send_action(action)
    np.testing.assert_allclose(robot.ros2_interface.pose_calls[-1], ROBOT_POSE, atol=1e-12)
    assert list(sent) == list(action)
    np.testing.assert_allclose(list(sent.values()), DATA_FRAME_POSE, atol=1e-12)


def test_reports_limited_pose_in_data_frame(robot):
    # Interface limit halves the robot-frame x: -0.4 -> -0.2, i.e. +0.2 in the data frame
    robot.ros2_interface.pose_result = lambda pose: np.asarray(pose) * [0.5, 1, 1, 1, 1, 1, 1, 1, 1]
    sent = robot.send_action({f"target_robot_pose.{n}.pos": v for n, v in enumerate(DATA_FRAME_POSE)})
    assert sent["target_robot_pose.0.pos"] == pytest.approx(0.2)


def test_observation_replay_returns_to_start(robot):
    robot.send_action(robot.get_observation())
    np.testing.assert_allclose(robot.ros2_interface.pose_calls[-1], ROBOT_POSE, atol=1e-12)


def test_non_finite_pose_dropped(robot):
    action = {f"target_robot_pose.{n}.pos": v for n, v in enumerate(DATA_FRAME_POSE)}
    action["target_robot_pose.4.pos"] = float("nan")
    robot.send_action(action)
    assert robot.ros2_interface.pose_calls == []


DATA_FRAME_ROT = R.from_euler("xyz", [0.1, 0.2, 0.3])
ROBOT_ROT = R.from_euler("z", np.pi).inv() * DATA_FRAME_ROT


@pytest.fixture
def quat_robot(tmp_path):
    config = CartesianRobotSimNoMiddleConfig(
        cameras={}, calibration_dir=tmp_path, observation_names=[f"current_robot_pose.{n}.pos" for n in range(7)]
    )
    config.ros2_interface.use_6d_rotation = False
    robot = CartesianRobot(config)
    robot.ros2_interface = FakeInterface(robot.config.ros2_interface)
    robot_pose = [*ROBOT_POSE[:3], *ROBOT_ROT.as_quat()]
    robot.ros2_interface.joint_state["cart_pose"] = dict(zip(POSE_QUAT_KEYS, robot_pose, strict=True))
    return robot


def test_rotate_pose_z_quaternion_matches_matrix():
    quat = DATA_FRAME_ROT.as_quat()
    rotated = rotate_pose_z([0.4, 0.1, 0.3, *quat], 37.0)
    expected = R.from_euler("z", 37.0, degrees=True) * DATA_FRAME_ROT
    assert R.from_quat(rotated[3:]).approx_equal(expected)
    np.testing.assert_allclose(rotate_pose_z(rotated, 37.0, inverse=True)[3:], quat, atol=1e-12)
    # Invalid quaternions are passed on unchanged, for the interface to drop
    assert not np.any(rotate_pose_z([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], 37.0)[3:])


def test_quaternion_observation_in_data_frame(quat_robot):
    obs = quat_robot.get_observation()
    assert list(obs) == [f"current_robot_pose.{n}.pos" for n in range(7)]
    values = list(obs.values())
    np.testing.assert_allclose(values[:3], DATA_FRAME_POSE[:3], atol=1e-12)
    assert R.from_quat(values[3:]).approx_equal(DATA_FRAME_ROT)


def test_quaternion_action_sent_in_robot_frame(quat_robot):
    assert list(quat_robot.action_features) == [f"target_robot_pose.{n}.pos" for n in range(7)]
    action = dict(zip(quat_robot.action_features, [*DATA_FRAME_POSE[:3], *DATA_FRAME_ROT.as_quat()], strict=True))
    sent = quat_robot.send_action(action)
    np.testing.assert_allclose(quat_robot.ros2_interface.pose_calls[-1][:3], ROBOT_POSE[:3], atol=1e-12)
    assert R.from_quat(quat_robot.ros2_interface.pose_calls[-1][3:]).approx_equal(ROBOT_ROT)
    np.testing.assert_allclose(list(sent.values()), list(action.values()), atol=1e-12)


def test_quaternion_observation_replay_returns_to_start(quat_robot):
    quat_robot.send_action(quat_robot.get_observation())
    assert R.from_quat(quat_robot.ros2_interface.pose_calls[-1][3:]).approx_equal(ROBOT_ROT)
