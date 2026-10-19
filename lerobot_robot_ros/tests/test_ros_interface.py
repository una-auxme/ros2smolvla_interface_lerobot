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

"""Tests for ROS2Interface callbacks, invoked directly (no executor or ROS graph needed)."""

import threading

from conftest import ARM_JOINTS
from sensor_msgs.msg import JointState

from lerobot_robot_ros.config import ActionType, GripperActionType, ROS2InterfaceConfig
from lerobot_robot_ros.ros_interface import ROS2Interface


def make_interface(**overrides):
    config = ROS2InterfaceConfig(arm_joint_names=list(ARM_JOINTS), **overrides)
    return ROS2Interface(config, ActionType.CARTESIAN_VELOCITY_TWIST_MSG)


def joint_msg(names, position, velocity=None):
    return JointState(name=names, position=position, velocity=velocity if velocity is not None else [])


def test_full_message():
    iface = make_interface()
    iface._joint_state_callback(joint_msg(ARM_JOINTS + ["gripper_joint"], [0.1] * 7, [0.5] * 7))
    assert iface.has_full_joint_state()
    assert iface.joint_state["position"]["joint_3.pos"] == 0.1
    assert iface.joint_state["velocity"]["gripper_joint.pos"] == 0.5


def test_empty_velocity_does_not_raise():
    iface = make_interface()
    iface._joint_state_callback(joint_msg(ARM_JOINTS + ["gripper_joint"], [0.2] * 7))
    assert iface.joint_state["position"]["joint_1.pos"] == 0.2
    assert iface.joint_state["velocity"]["joint_1.pos"] == 0.0


def test_partial_messages_are_merged():
    iface = make_interface()
    iface._joint_state_callback(joint_msg(ARM_JOINTS, [0.1] * 6))
    assert not iface.has_full_joint_state()
    iface._joint_state_callback(joint_msg(["gripper_joint", "unrelated_joint"], [0.02, 9.0]))
    assert iface.has_full_joint_state()
    assert iface.joint_state["position"]["joint_6.pos"] == 0.1
    assert iface.joint_state["position"]["gripper_joint.pos"] == 0.02
    assert "unrelated_joint.pos" not in iface.joint_state["position"]


def test_gripper_not_required_without_gripper():
    iface = make_interface(gripper_action_type=GripperActionType.NONE)
    iface._joint_state_callback(joint_msg(ARM_JOINTS, [0.1] * 6))
    assert iface.has_full_joint_state()


def test_snapshot_not_mutated_by_later_messages():
    iface = make_interface()
    iface._joint_state_callback(joint_msg(ARM_JOINTS, [0.1] * 6))
    snapshot = iface.joint_state["position"]
    iface._joint_state_callback(joint_msg(ARM_JOINTS, [0.2] * 6))
    assert snapshot["joint_1.pos"] == 0.1


def test_not_connected_when_spin_thread_died():
    iface = make_interface()
    assert not iface.is_connected
    iface.executor_thread = threading.Thread(target=lambda: None)
    iface.executor_thread.start()
    iface.executor_thread.join()
    iface.is_connected = True
    assert not iface.is_connected


def test_state_age():
    iface = make_interface()
    assert iface.state_age("position") is None
    iface._joint_state_callback(joint_msg(ARM_JOINTS, [0.1] * 6))
    assert 0.0 <= iface.state_age("position") < 0.5
    assert iface.state_age("cart_pose") is None
