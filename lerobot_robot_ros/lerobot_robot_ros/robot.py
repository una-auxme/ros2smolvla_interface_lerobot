# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import ActionType, GripperActionType, ROS2Config
from .ros_interface import ROS2Interface

logger = logging.getLogger(__name__)

GRIPPER_ACTION_KEY = "gripper.pos"
# Gripper actions are normalized: 0 = open, 1 = closed
GRIPPER_CLOSE_THRESHOLD = 0.5


class ROS2Robot(Robot):
    config_class = ROS2Config
    name = "ros2"

    def __init__(self, config: ROS2Config):
        super().__init__(config)
        self.config = config
        self.ros2_interface = ROS2Interface(config.ros2_interface, config.action_type)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.last_gripper_action = None

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        
        state_ft = {obs: float for obs in self.config.observation_names}        
        return {**state_ft, **self._cameras_ft}

    @property
    def _has_gripper(self) -> bool:
        return self.config.ros2_interface.gripper_action_type is not GripperActionType.NONE

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.action_type in (ActionType.CARTESIAN_VELOCITY, ActionType.CARTESIAN_VELOCITY_TWIST_MSG):
            arm_ft = {
                "linear_x.vel": float,
                "linear_y.vel": float,
                "linear_z.vel": float,
                "angular_x.vel": float,
                "angular_y.vel": float,
                "angular_z.vel": float,
            }
        elif self.config.action_type in (ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY):
            arm_ft = {f"{joint}.pos": float for joint in self.config.ros2_interface.arm_joint_names}
        else:
            raise ValueError(f"Unsupported action type: {self.config.action_type}")
        return arm_ft | ({GRIPPER_ACTION_KEY: float} if self._has_gripper else {})

    @property
    def is_connected(self) -> bool:
        return self.ros2_interface.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for cam in self.cameras.values():
            cam.connect()
        self.ros2_interface.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass  # robot must be calibrated before running LeRobot

    def configure(self) -> None:
        pass  # robot must be configured before running LeRobot

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}
        joint_state = self.ros2_interface.joint_state
        obs_dict.update({
            joint : pos 
            for joint, pos in joint_state["position"].items() 
            if joint in self.config.observation_names
        })

        obs_dict.update({
            joint : pos 
            for joint, pos in joint_state["cart_pose"].items() 
            if joint in self.config.observation_names
        })

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            try:
                obs_dict[cam_key] = cam.async_read(timeout_ms=300)
            except Exception as e:
                logger.error(f"Failed to read camera {cam_key}: {e}")
                obs_dict[cam_key] = None
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors or pressed_keys dict.

        Raises:
            DeviceNotConnectedError: if robot is not connected.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = dict(action)
        if self.config.action_type in (ActionType.CARTESIAN_VELOCITY, ActionType.CARTESIAN_VELOCITY_TWIST_MSG):
            if self.config.max_relative_target is not None:
                # We don't have the current velocity of the arm, so set it to 0.0
                # Effectively the goal velocity gets clipped by max_relative_target
                goal_present_vel = {
                    key: (act, 0.0) for key, act in action.items() if key != GRIPPER_ACTION_KEY
                }
                action.update(self._clip_relative(goal_present_vel))

            linear_vel = (
                action["linear_x.vel"],
                action["linear_y.vel"],
                action["linear_z.vel"],
            )
            angular_vel = (
                action["angular_x.vel"],
                action["angular_y.vel"],
                action["angular_z.vel"],
            )
            self.ros2_interface.servo(linear=linear_vel, angular=angular_vel)
        elif self.config.action_type in (ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY):
            if self.config.max_relative_target is not None:
                joint_state = self.ros2_interface.joint_state
                if joint_state is None or "position" not in joint_state:
                    raise ValueError("Joint state is not available yet.")

                goal_present_pos = {}
                for joint in self.config.ros2_interface.arm_joint_names:
                    key = f"{joint}.pos"
                    if key not in joint_state["position"]:
                        raise ValueError(f"No present position for '{joint}', cannot apply max_relative_target.")
                    goal_present_pos[key] = (action[key], joint_state["position"][key])
                action.update(self._clip_relative(goal_present_pos))

            joint_positions = [action[joint + ".pos"] for joint in self.config.ros2_interface.arm_joint_names]
            self.ros2_interface.send_joint_position_command(joint_positions)

        if self._has_gripper and GRIPPER_ACTION_KEY in action:
            if action[GRIPPER_ACTION_KEY] > GRIPPER_CLOSE_THRESHOLD:
                gripper_pos = self.config.ros2_interface.gripper_close_position
            else:
                gripper_pos = self.config.ros2_interface.gripper_open_position

            # Only send on change; remember it only once it went out, so a failed send is retried
            if gripper_pos != self.last_gripper_action and self.ros2_interface.send_gripper_command(gripper_pos):
                self.last_gripper_action = gripper_pos
        return action

    def _clip_relative(self, goal_present: dict[str, tuple[float, float]]) -> dict[str, float]:
        """Clip goals to within `max_relative_target` of their present value."""
        max_relative_target = self.config.max_relative_target
        if isinstance(max_relative_target, int):
            # ensure_safe_goal_position only accepts a float or a dict
            max_relative_target = float(max_relative_target)
        elif isinstance(max_relative_target, dict):
            max_relative_target = {key: max_relative_target[key] for key in goal_present}
        return ensure_safe_goal_position(goal_present, max_relative_target)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()
        self.ros2_interface.disconnect()

        logger.info(f"{self} disconnected.")


class SO101ROS(ROS2Robot):
    pass


class AnninAR4(ROS2Robot):
    pass

class UR10eSim(ROS2Robot):
    pass

class UR10eReal(ROS2Robot):
    pass