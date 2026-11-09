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

from lerobot_robot_ros.config import (
    ActionType,
    CartesianRobotRealConfig,
    CartesianRobotRealNoMiddleConfig,
    CartesianRobotSimConfig,
    CartesianRobotSimNoMiddleConfig,
)
from lerobot_robot_ros.robot import (
    OFFSET_ACTION_KEYS,
    CartesianRobot,
    apply_pose_offset,
    pose_offset,
    rotate_pose_z,
)

DATA_FRAME_POSE = [0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
ROBOT_POSE = [-0.4, -0.1, 0.3, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]  # DATA_FRAME_POSE rotated by 180 deg about z


@pytest.fixture
def robot(tmp_path):
    # Absolute poses and measured states, the configs' defaults are tested below
    return make_cartesian_robot(tmp_path, zero_state=False, offset_actions=False)


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
        cameras={}, calibration_dir=tmp_path, observation_names=[f"current_robot_pose.{n}.pos" for n in range(7)],
        zero_state=False, offset_actions=False,
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


OFFSET = [0.01, -0.02, 0.03, 0.05, -0.1, 0.2]


def make_cartesian_robot(tmp_path, **overrides):
    robot = CartesianRobot(CartesianRobotSimNoMiddleConfig(cameras={}, calibration_dir=tmp_path, **overrides))
    robot.ros2_interface = FakeInterface(robot.config.ros2_interface)
    robot.ros2_interface.joint_state["cart_pose"] = dict(zip(POSE_KEYS, ROBOT_POSE, strict=True))
    return robot


@pytest.fixture
def offset_robot(tmp_path):
    return make_cartesian_robot(tmp_path)


@pytest.mark.parametrize(
    "config_class",
    [CartesianRobotRealConfig, CartesianRobotSimConfig, CartesianRobotSimNoMiddleConfig, CartesianRobotRealNoMiddleConfig],
)
def test_cartesian_configs_default_to_zero_state_and_offsets(config_class):
    config = config_class()
    assert config.zero_state and config.offset_actions


def offset_action(offset):
    return dict(zip(OFFSET_ACTION_KEYS, offset, strict=True))


def expected_target(pose, offset):
    """Data frame target of the runner's apply_offsets: p + dp, R_offset * R."""
    rotation = R.from_rotvec(offset[3:]).as_matrix() @ np.column_stack((pose[3:6], pose[6:9], np.cross(pose[3:6], pose[6:9])))
    return np.concatenate((np.add(pose[:3], offset[:3]), rotation[:, 0], rotation[:, 1]))


def test_zero_state_observation(offset_robot):
    obs = offset_robot.get_observation()
    assert list(obs) == [f"current_robot_pose.{n}.pos" for n in range(9)]
    assert all(value == 0.0 for value in obs.values())


def test_offset_action_features(offset_robot):
    assert list(offset_robot.action_features) == OFFSET_ACTION_KEYS


def test_offset_applied_to_measured_pose_in_data_frame(offset_robot):
    sent = offset_robot.send_action(offset_action(OFFSET))
    target = expected_target(DATA_FRAME_POSE, OFFSET)
    np.testing.assert_allclose(offset_robot.ros2_interface.pose_calls[-1], rotate_pose_z(target, 180.0, inverse=True),
                               atol=1e-12)
    assert list(sent) == OFFSET_ACTION_KEYS
    np.testing.assert_allclose(list(sent.values()), OFFSET, atol=1e-12)


def test_offset_follows_the_measured_pose(offset_robot):
    offset_robot.send_action(offset_action([0.0] * 6))
    np.testing.assert_allclose(offset_robot.ros2_interface.pose_calls[-1], ROBOT_POSE, atol=1e-12)
    moved = [-0.5, -0.1, 0.3, *ROBOT_POSE[3:]]
    offset_robot.ros2_interface.joint_state["cart_pose"] = dict(zip(POSE_KEYS, moved, strict=True))
    offset_robot.send_action(offset_action([0.0] * 6))
    np.testing.assert_allclose(offset_robot.ros2_interface.pose_calls[-1], moved, atol=1e-12)


def test_offset_reports_limited_offset(offset_robot):
    # The interface pulls the robot-frame x back to the measured -0.4: no x offset is applied in the data frame
    offset_robot.ros2_interface.pose_result = lambda pose: np.concatenate(([-0.4], pose[1:]))
    sent = offset_robot.send_action(offset_action(OFFSET))
    assert sent["offset_robot_pose.0.pos"] == pytest.approx(0.0)
    np.testing.assert_allclose(list(sent.values())[1:], OFFSET[1:], atol=1e-12)


def test_offset_non_finite_dropped(offset_robot):
    offset_robot.send_action(offset_action([0.0, float("nan"), 0.0, 0.0, 0.0, 0.0]))
    assert offset_robot.ros2_interface.pose_calls == []


def test_offset_without_measured_pose_dropped(offset_robot):
    del offset_robot.ros2_interface.joint_state["cart_pose"]
    offset_robot.send_action(offset_action(OFFSET))
    assert offset_robot.ros2_interface.pose_calls == []


def test_zero_state_refuses_observation_replay(offset_robot):
    # lerobot-rollout's return to the initial position would replay the zeroed observation
    with pytest.raises(ValueError, match="zero_state"):
        offset_robot.send_action(offset_robot.get_observation())
    assert offset_robot.ros2_interface.pose_calls == []


def test_offset_robot_accepts_absolute_targets(offset_robot):
    offset_robot.send_action({f"target_robot_pose.{n}.pos": v for n, v in enumerate(DATA_FRAME_POSE)})
    np.testing.assert_allclose(offset_robot.ros2_interface.pose_calls[-1], ROBOT_POSE, atol=1e-12)


def test_zero_state_with_absolute_actions(tmp_path):
    robot = make_cartesian_robot(tmp_path, offset_actions=False)
    assert list(robot.action_features) == [f"target_robot_pose.{n}.pos" for n in range(9)]
    assert not any(robot.get_observation().values())


def test_offset_needs_cartesian_position(tmp_path):
    with pytest.raises(ValueError, match="cartesian_position"):
        make_cartesian_robot(tmp_path, offset_actions=True, action_type=ActionType.JOINT_POSITION)


@pytest.mark.parametrize("flag", ["zero_state", "offset_actions"])
def test_flags_need_cartesian_robot(make_robot, flag):
    with pytest.raises(ValueError, match="CartesianRobot"):
        make_robot(ActionType.CARTESIAN_POSITION, **{flag: True})


def test_pose_offset_roundtrip_quaternion():
    pose = [0.4, 0.1, 0.3, *DATA_FRAME_ROT.as_quat()]
    target = apply_pose_offset(pose, OFFSET)
    np.testing.assert_allclose(target[:3], np.add(pose[:3], OFFSET[:3]), atol=1e-12)
    assert R.from_quat(target[3:]).approx_equal(R.from_rotvec(OFFSET[3:]) * DATA_FRAME_ROT)
    np.testing.assert_allclose(pose_offset(pose, target), OFFSET, atol=1e-12)


def test_quaternion_offset_action(tmp_path):
    config = CartesianRobotSimNoMiddleConfig(
        cameras={}, calibration_dir=tmp_path,
        observation_names=[f"current_robot_pose.{n}.pos" for n in range(7)],
    )
    config.ros2_interface.use_6d_rotation = False
    robot = CartesianRobot(config)
    robot.ros2_interface = FakeInterface(robot.config.ros2_interface)
    robot.ros2_interface.joint_state["cart_pose"] = dict(zip(POSE_QUAT_KEYS, [*ROBOT_POSE[:3], *ROBOT_ROT.as_quat()],
                                                             strict=True))
    sent = robot.send_action(offset_action(OFFSET))
    sent_pose = rotate_pose_z(robot.ros2_interface.pose_calls[-1], 180.0)
    np.testing.assert_allclose(sent_pose[:3], np.add(DATA_FRAME_POSE[:3], OFFSET[:3]), atol=1e-12)
    assert R.from_quat(sent_pose[3:]).approx_equal(R.from_rotvec(OFFSET[3:]) * DATA_FRAME_ROT)
    np.testing.assert_allclose(list(sent.values()), OFFSET, atol=1e-12)
