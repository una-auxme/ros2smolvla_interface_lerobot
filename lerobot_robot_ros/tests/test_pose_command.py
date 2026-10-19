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

"""Tests for ROS2Interface.send_cartesian_position_command with a stubbed publisher."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ARM_JOINTS
from geometry_msgs.msg import PoseStamped
from rclpy.clock import Clock
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ros import ros_interface as ros_interface_module
from lerobot_robot_ros.config import ActionType, GripperActionType, ROS2InterfaceConfig
from lerobot_robot_ros.ros_interface import ROS2Interface

IDENTITY_6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class StubPublisher:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


def make_interface(**overrides):
    config = ROS2InterfaceConfig(
        arm_joint_names=list(ARM_JOINTS),
        gripper_action_type=GripperActionType.NONE,
        use_6d_rotation=True,
        **overrides,
    )
    iface = ROS2Interface(config, ActionType.CARTESIAN_POSITION)
    iface.robot_node = SimpleNamespace(get_clock=Clock)
    iface.pose_cmd_pub = StubPublisher()
    return iface


def pose_msg(xyz=(0.0, 0.0, 0.3), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="base_link"):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = xyz
    o = msg.pose.orientation
    o.x, o.y, o.z, o.w = quat_xyzw
    return msg


def sent_position(iface):
    p = iface.pose_cmd_pub.msgs[-1].pose.position
    return [p.x, p.y, p.z]


def sent_rotation(iface):
    o = iface.pose_cmd_pub.msgs[-1].pose.orientation
    return R.from_quat([o.x, o.y, o.z, o.w])


def test_unlimited_pose_is_sent_as_is():
    iface = make_interface()
    rot = R.from_euler("xyz", [0.1, 0.2, 0.3])
    m = rot.as_matrix()
    assert iface.send_cartesian_position_command([0.4, -0.1, 0.5, *m[:, 0], *m[:, 1]])
    msg = iface.pose_cmd_pub.msgs[-1]
    assert msg.header.frame_id == "base_link"
    np.testing.assert_allclose(sent_position(iface), [0.4, -0.1, 0.5])
    assert sent_rotation(iface).approx_equal(rot)


@pytest.mark.parametrize(
    "pose",
    [[float("nan")] + [0.0] * 8, [0.0] * 9, [0.0, 0.0, 0.0, 1, 0, 0, 2, 0, 0], [0.0] * 6],
)
def test_invalid_pose_dropped(pose):
    iface = make_interface()
    assert not iface.send_cartesian_position_command(pose)
    assert iface.pose_cmd_pub.msgs == []


def test_workspace_and_step_limits():
    iface = make_interface(workspace_min=(-1.0, -1.0, 0.25), workspace_max=(1.0, 1.0, 1.0), max_cartesian_step=0.02)
    iface._cart_pose_callback(pose_msg(xyz=(0.0, 0.0, 0.3)))
    assert iface.send_cartesian_position_command([0.0, 0.0, 0.0, *IDENTITY_6D])
    np.testing.assert_allclose(sent_position(iface), [0.0, 0.0, 0.28])


def test_rotation_step_limit():
    iface = make_interface(max_rotation_step=0.1)
    iface._cart_pose_callback(pose_msg())
    m = R.from_euler("z", np.pi / 2).as_matrix()
    assert iface.send_cartesian_position_command([0.0, 0.0, 0.3, *m[:, 0], *m[:, 1]])
    np.testing.assert_allclose(sent_rotation(iface).as_rotvec(), [0.0, 0.0, 0.1], atol=1e-9)


@pytest.mark.parametrize("pose_age", [None, 5.0])
def test_step_limit_needs_fresh_pose(pose_age):
    iface = make_interface(max_cartesian_step=0.02)
    if pose_age is not None:
        iface._cart_pose_callback(pose_msg())
        iface._state_stamps["cart_pose"] = time.monotonic() - pose_age
    assert not iface.send_cartesian_position_command([0.0, 0.0, 0.3, *IDENTITY_6D])
    assert iface.pose_cmd_pub.msgs == []


def test_pose_observation_6d_and_quaternion():
    rot = R.from_euler("xyz", [0.4, -0.2, 1.0])
    quat = rot.as_quat()
    iface = make_interface()
    iface._cart_pose_callback(pose_msg(quat_xyzw=quat))
    cart = iface.joint_state["cart_pose"]
    m = rot.as_matrix()
    np.testing.assert_allclose(
        [cart[k] for k in ["pose.r11", "pose.r21", "pose.r31", "pose.r12", "pose.r22", "pose.r32"]],
        [*m[:, 0], *m[:, 1]],
    )

    quat_iface = ROS2Interface(ROS2InterfaceConfig(), ActionType.CARTESIAN_VELOCITY_TWIST_MSG)
    quat_iface._cart_pose_callback(pose_msg(quat_xyzw=quat))
    assert [quat_iface.joint_state["cart_pose"][f"pose.quat_{k}"] for k in "xyzw"] == list(quat)


def test_invalid_orientation_ignored():
    iface = make_interface()
    iface._cart_pose_callback(pose_msg(quat_xyzw=(0.0, 0.0, 0.0, 0.0)))
    assert iface.joint_state is None
    assert iface.state_age("cart_pose") is None


def test_frame_mismatch_warns_once(monkeypatch):
    messages = []
    monkeypatch.setattr(ros_interface_module.logger, "warning", messages.append)
    iface = make_interface(base_link="base_link")
    for _ in range(3):
        iface._cart_pose_callback(pose_msg(frame_id="world"))
    assert len(messages) == 1
    assert "'world'" in messages[0]
