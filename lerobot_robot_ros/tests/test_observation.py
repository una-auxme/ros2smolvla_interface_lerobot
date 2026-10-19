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
from conftest import ARM_JOINTS

from lerobot_robot_ros import robot as robot_module
from lerobot_robot_ros.config import ActionType

JOINT_OBS = [f"{j}.pos" for j in ARM_JOINTS]


@pytest.mark.parametrize("action_type", list(ActionType))
def test_joint_observation_without_cartesian_pose(make_robot, action_type):
    robot = make_robot(action_type=action_type, observation_names=JOINT_OBS)
    assert robot.get_observation() == dict.fromkeys(JOINT_OBS, 0.0)


def test_cartesian_pose_observation(make_robot):
    robot = make_robot(observation_names=["joint_1.pos", "pose.x", "pose.quat_w"])
    robot.ros2_interface.joint_state["cart_pose"] = {"pose.x": 0.4, "pose.quat_w": 1.0, "pose.y": 0.0}
    assert robot.get_observation() == {"joint_1.pos": 0.0, "pose.x": 0.4, "pose.quat_w": 1.0}


def test_no_state_yet_raises_clear_error(make_robot):
    robot = make_robot()
    robot.ros2_interface.joint_state = None
    with pytest.raises(RuntimeError, match="No joint state"):
        robot.get_observation()


def test_missing_and_stale_state_warn_once(make_robot, monkeypatch):
    messages = []
    monkeypatch.setattr(robot_module.logger, "warning", messages.append)
    robot = make_robot(observation_names=JOINT_OBS + ["pose.x"])
    robot.ros2_interface.state_age = lambda kind: 2.0 if kind == "position" else None
    for _ in range(3):
        robot.get_observation()
    assert sum("stale" in m for m in messages) == 1
    assert sum("missing ['pose.x']" in m for m in messages) == 1
