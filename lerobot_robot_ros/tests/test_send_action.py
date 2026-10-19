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

import pytest
from conftest import joint_action, velocity_action

from lerobot_robot_ros.config import ActionType


def test_velocity_unclipped_by_default(make_robot):
    robot = make_robot()
    robot.send_action(velocity_action(0.9))
    assert robot.ros2_interface.servo_calls == [((0.9,) * 3, (0.9,) * 3)]


@pytest.mark.parametrize("limit", [0.2, {f"{k}.vel": 0.2 for k in
                                         ["linear_x", "linear_y", "linear_z",
                                          "angular_x", "angular_y", "angular_z"]}])
def test_velocity_clipped_but_gripper_untouched(make_robot, limit):
    robot = make_robot(max_relative_target=limit)
    sent = robot.send_action(velocity_action(-0.9, gripper=1.0))
    assert robot.ros2_interface.servo_calls == [((-0.2,) * 3, (-0.2,) * 3)]
    assert sent["gripper.pos"] == 1.0


def test_int_max_relative_target_accepted(make_robot):
    robot = make_robot(max_relative_target=1)
    robot.send_action(velocity_action(3.0))
    assert robot.ros2_interface.servo_calls == [((1.0,) * 3, (1.0,) * 3)]


def test_joint_clip_is_relative_to_present_position(make_robot):
    robot = make_robot(action_type=ActionType.JOINT_POSITION, max_relative_target=0.1)
    present = robot.ros2_interface.joint_state["position"]
    for key in present:
        present[key] = 1.0
    robot.send_action(joint_action(pos=1.05, gripper=1.0))
    assert robot.ros2_interface.joint_calls == [[1.05] * 6]

    robot.send_action(joint_action(pos=2.0))
    assert robot.ros2_interface.joint_calls[-1] == pytest.approx([1.1] * 6)


def test_joint_clip_requires_joint_state(make_robot):
    robot = make_robot(action_type=ActionType.JOINT_POSITION, max_relative_target=0.1)
    robot.ros2_interface.joint_state = None
    with pytest.raises(ValueError):
        robot.send_action(joint_action())
    assert robot.ros2_interface.joint_calls == []


@pytest.mark.parametrize(
    ("open_pos", "close_pos"),
    [(0.0249, 0.0001), (0.0, 1.0), (1.0, 0.0)],
)
@pytest.mark.parametrize(("command", "expect_closed"), [(0.0, False), (0.3, False), (0.7, True), (1.0, True)])
def test_gripper_threshold_is_normalized(make_robot, open_pos, close_pos, command, expect_closed):
    robot = make_robot(ros2_interface={"gripper_open_position": open_pos, "gripper_close_position": close_pos})
    robot.send_action(velocity_action(gripper=command))
    assert robot.ros2_interface.gripper_calls == [close_pos if expect_closed else open_pos]


def test_gripper_only_sent_on_change(make_robot):
    robot = make_robot()
    for command in [0.0, 0.0, 1.0, 1.0, 0.0]:
        robot.send_action(velocity_action(gripper=command))
    assert robot.ros2_interface.gripper_calls == [0.0, 1.0, 0.0]


def test_failed_gripper_command_is_retried(make_robot):
    robot = make_robot()
    robot.ros2_interface.send_gripper_command = lambda pos: robot.ros2_interface.gripper_calls.append(pos)
    robot.send_action(velocity_action(gripper=1.0))
    robot.send_action(velocity_action(gripper=1.0))
    assert robot.ros2_interface.gripper_calls == [1.0, 1.0]


def test_action_without_gripper_key(make_robot):
    robot = make_robot()
    action = velocity_action()
    del action["gripper.pos"]
    robot.send_action(action)
    assert robot.ros2_interface.gripper_calls == []


def test_no_gripper(make_robot):
    from lerobot_robot_ros.config import GripperActionType

    robot = make_robot(ros2_interface={"gripper_action_type": GripperActionType.NONE})
    assert "gripper.pos" not in robot.action_features
    robot.send_action(velocity_action(gripper=1.0))
    assert robot.ros2_interface.gripper_calls == []


def test_action_features_include_gripper_by_default(make_robot):
    assert list(make_robot().action_features)[-1] == "gripper.pos"
    joint_robot = make_robot(action_type=ActionType.JOINT_POSITION)
    assert list(joint_robot.action_features) == [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
