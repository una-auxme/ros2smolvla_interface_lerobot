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
import math
import time
from functools import cached_property
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import ActionType, GripperActionType, ROS2Config
from .ros_interface import POSE_6D_KEYS, POSE_QUAT_KEYS, ROS2Interface
from .safety import rotation_from_6d, rotation_from_quat

logger = logging.getLogger(__name__)

GRIPPER_ACTION_KEY = "gripper.pos"
# Gripper actions are normalized: 0 = open, 1 = closed
GRIPPER_CLOSE_THRESHOLD = 0.5


class ROS2Robot(Robot):
    config_class = ROS2Config
    name = "ros2"

    def __init__(self, config: ROS2Config):
        super().__init__(config)
        if (config.zero_state or config.offset_actions) and not isinstance(self, CartesianRobot):
            raise ValueError("zero_state and offset_actions are only supported by CartesianRobot (cartesian_robot_*)")
        self.config = config
        self.ros2_interface = ROS2Interface(config.ros2_interface, config.action_type)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.last_gripper_action = None
        self._last_warning_time: dict[str, float] = {}

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

    @property
    def _pose_keys(self) -> list[str]:
        """Keys of a Cartesian pose action, the same as the observed pose."""
        return POSE_6D_KEYS if self.config.ros2_interface.use_6d_rotation else POSE_QUAT_KEYS

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
        elif self.config.action_type is ActionType.CARTESIAN_POSITION:
            # Same keys as the observed pose, so an observation can be replayed as an action
            arm_ft = dict.fromkeys(self._pose_keys, float)
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

        state = self._read_state()
        obs_dict: dict[str, Any] = {name: state[name] for name in self.config.observation_names if name in state}
        missing = [name for name in self.config.observation_names if name not in obs_dict]
        if missing:
            self._warn_throttled("missing_obs", f"Observation is missing {missing}. Are they published?")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = self._read_camera(cam_key, cam)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def _read_state(self) -> dict[str, float]:
        """All robot state values currently available, e.g. "shoulder_pan_joint.pos" or "pose.x"."""
        joint_state = self.ros2_interface.joint_state
        if joint_state is None:
            raise RuntimeError("No joint state / Cartesian pose received yet. Is the controller running?")

        state: dict[str, float] = {}
        # Only Cartesian control subscribes to the Cartesian pose, so it may be absent
        for kind in ("position", "cart_pose"):
            state.update(joint_state.get(kind, {}))
            age = self.ros2_interface.state_age(kind)
            if age is not None and age > self.config.ros2_interface.max_state_age_s:
                self._warn_throttled(f"stale_{kind}", f"Observed {kind} is {age:.2f} s old, the robot state may be stale.")
        return state

    def _read_camera(self, cam_key: str, cam) -> Any:
        try:
            return cam.async_read(timeout_ms=300)
        except Exception as e:
            logger.error(f"Failed to read camera {cam_key}: {e}")
            return None

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors or pressed_keys dict.

        Actions containing NaN or infinite values are dropped entirely and nothing is sent.

        Raises:
            DeviceNotConnectedError: if robot is not connected.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Drop the whole action, gripper included: a NaN gripper command would fail the close
        # threshold comparison and open the gripper.
        non_finite = [key for key, value in action.items() if not math.isfinite(float(value))]
        if non_finite:
            logger.error(f"Dropping action with non-finite values for {non_finite}: {action}")
            return action

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
        elif self.config.action_type is ActionType.CARTESIAN_POSITION:
            sent = self.ros2_interface.send_cartesian_position_command([action[key] for key in self._pose_keys])
            if sent is not None:
                # Report the pose after workspace / step limits
                action.update(zip(self._pose_keys, sent.tolist(), strict=True))

        if self._has_gripper and GRIPPER_ACTION_KEY in action:
            if action[GRIPPER_ACTION_KEY] > GRIPPER_CLOSE_THRESHOLD:
                gripper_pos = self.config.ros2_interface.gripper_close_position
            else:
                gripper_pos = self.config.ros2_interface.gripper_open_position

            # Only send on change; remember it only once it went out, so a failed send is retried
            if gripper_pos != self.last_gripper_action and self.ros2_interface.send_gripper_command(gripper_pos):
                self.last_gripper_action = gripper_pos
        return action

    def _warn_throttled(self, key: str, message: str, period_s: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._last_warning_time.get(key, -period_s) >= period_s:
            self._last_warning_time[key] = now
            logger.warning(message)

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


# CartesianRobot velocity feature names
_CARTESIAN_VEL_KEYS = [f"delta_robot_pose.{n}.vel" for n in range(6)]
_ROS2_VEL_KEYS = ["linear_x.vel", "linear_y.vel", "linear_z.vel", "angular_x.vel", "angular_y.vel", "angular_z.vel"]
# CartesianRobot offset actions (`offset_actions`): dx, dy, dz, then the rotation vector. lerobot-rollout only maps
# policy outputs onto action features ending in .pos or .vel
OFFSET_ACTION_KEYS = [f"offset_robot_pose.{n}.pos" for n in range(6)]


def rotate_pose_z(pose, yaw_deg: float, inverse: bool = False) -> np.ndarray:
    """Rotate a pose about the z axis: 9-D (xyz + first two rotation-matrix columns) or 7-D (xyz + an
    (x, y, z, w) quaternion).

    Applies p' = Rz(yaw) p to the position and R' = Rz(yaw) R to the orientation. With `inverse=True`
    the opposite rotation is applied. A quaternion is not normalized, so invalid ones stay invalid.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.shape not in ((7,), (9,)):
        raise ValueError(f"Expected a 7-D or 9-D pose, got shape {pose.shape}")
    if yaw_deg == 0.0:
        return pose.copy()
    yaw = np.deg2rad(-yaw_deg if inverse else yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if pose.shape == (9,):
        return np.concatenate([rz @ pose[0:3], rz @ pose[3:6], rz @ pose[6:9]])
    # Hamilton product q_z * q with q_z = (0, 0, sin(yaw / 2), cos(yaw / 2))
    x, y, z, w = pose[3:7]
    hc, hs = np.cos(yaw / 2), np.sin(yaw / 2)
    quat = [hc * x - hs * y, hc * y + hs * x, hc * z + hs * w, hc * w - hs * z]
    return np.concatenate([rz @ pose[0:3], quat])


def _pose_rotation(pose: np.ndarray) -> R | None:
    """Orientation of a 9-D or 7-D pose (see rotate_pose_z), None if it is degenerate."""
    if pose.shape == (9,):
        matrix = rotation_from_6d(pose[3:6], pose[6:9])
        return None if matrix is None else R.from_matrix(matrix)
    return rotation_from_quat(pose[3:7])


def apply_pose_offset(pose, offset) -> np.ndarray | None:
    """The 9-D or 7-D pose moved by a 6-D offset (position difference + rotation vector, in the pose's frame):
    p' = p + dp, R' = R_offset * R. None if the pose's rotation is degenerate.

    The inverse of pose_offset, and the convention of the nora_pipeline_runner's offset_actions.py.
    """
    pose, offset = np.asarray(pose, dtype=float), np.asarray(offset, dtype=float)
    rotation = _pose_rotation(pose)
    if rotation is None:
        return None
    rotation = R.from_rotvec(offset[3:6]) * rotation
    if pose.shape == (9,):
        matrix = rotation.as_matrix()
        return np.concatenate([pose[:3] + offset[:3], matrix[:, 0], matrix[:, 1]])
    quat = rotation.as_quat()
    # q and -q are the same rotation: keep the sign closest to the pose's, so poses stay continuous
    return np.concatenate([pose[:3] + offset[:3], -quat if np.dot(quat, pose[3:7]) < 0 else quat])


def pose_offset(pose, target) -> np.ndarray | None:
    """The 6-D offset that moves a 9-D or 7-D pose to the target pose (see apply_pose_offset). None if either
    rotation is degenerate."""
    pose, target = np.asarray(pose, dtype=float), np.asarray(target, dtype=float)
    rotation, target_rotation = _pose_rotation(pose), _pose_rotation(target)
    if rotation is None or target_rotation is None:
        return None
    return np.concatenate([target[:3] - pose[:3], (target_rotation * rotation.inv()).as_rotvec()])


class CartesianRobot(ROS2Robot):
    """ROS2Robot with index-based Cartesian feature names, expressed in a configurable data frame.

    Observations: current_robot_pose.{n}.pos and current_joint_positions.{i}. Actions:
    target_robot_pose.{n}.pos (CARTESIAN_POSITION), delta_robot_pose.{0-5}.vel (velocity control) or
    target_joint_positions.{i}.pos (joint control). The pose is xyz + first two rotation-matrix columns
    (n = 0-8) with `use_6d_rotation`, otherwise xyz + an (x, y, z, w) quaternion (n = 0-6).

    The pose is rotated by `data_frame_yaw_deg` about z between `base_link` and the data frame, e.g.
    the frame a dataset was recorded or generated in. All limits and checks of ROS2Robot / ROS2Interface
    apply.

    With `zero_state` every observed state value is reported as 0. With `offset_actions` (CARTESIAN_POSITION
    only) the actions are offset_robot_pose.{0-5}.pos: position difference and rotation vector from the pose
    measured when the action is sent, in the data frame (see apply_pose_offset).
    """

    def __init__(self, config):
        super().__init__(config)
        if config.offset_actions and config.action_type is not ActionType.CARTESIAN_POSITION:
            raise ValueError(f"offset_actions needs action_type cartesian_position, not {config.action_type.value}")

    @property
    def _yaw_deg(self) -> float:
        return self.config.ros2_interface.data_frame_yaw_deg

    @property
    def _obs_pose_keys(self) -> list[str]:
        return [f"current_robot_pose.{n}.pos" for n in range(len(self._pose_keys))]

    @property
    def _target_pose_keys(self) -> list[str]:
        return [f"target_robot_pose.{n}.pos" for n in range(len(self._pose_keys))]

    def _action_key_map(self) -> dict[str, str]:
        """CartesianRobot action key -> ROS2Robot action key."""
        if self.config.action_type in (ActionType.CARTESIAN_VELOCITY, ActionType.CARTESIAN_VELOCITY_TWIST_MSG):
            return dict(zip(_CARTESIAN_VEL_KEYS, _ROS2_VEL_KEYS, strict=True))
        if self.config.action_type in (ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY):
            return {
                f"target_joint_positions.{i}.pos": f"{joint}.pos"
                for i, joint in enumerate(self.config.ros2_interface.arm_joint_names)
            }
        if self.config.action_type is ActionType.CARTESIAN_POSITION:
            return dict(zip(self._target_pose_keys, self._pose_keys, strict=True))
        raise ValueError(f"Unsupported action type: {self.config.action_type}")

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.offset_actions:
            return dict.fromkeys(OFFSET_ACTION_KEYS, float)
        return dict.fromkeys(self._action_key_map(), float)

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        if self.config.zero_state:
            obs.update(dict.fromkeys(self.config.observation_names, 0.0))
        return obs

    def _measured_pose(self) -> np.ndarray | None:
        """The current pose in the data frame (not zeroed by `zero_state`), None if it is not observed."""
        state = self._read_state()
        if not all(key in state for key in self._obs_pose_keys):
            return None
        return np.array([state[key] for key in self._obs_pose_keys])

    def _read_state(self) -> dict[str, float]:
        state = super()._read_state()
        cartesian_state = {
            f"current_joint_positions.{i}": state[f"{joint}.pos"]
            for i, joint in enumerate(self.config.ros2_interface.arm_joint_names)
            if f"{joint}.pos" in state
        }
        obs_pose_keys = self._obs_pose_keys
        cartesian_state.update(
            {name: state[key] for name, key in zip(obs_pose_keys, self._pose_keys, strict=True) if key in state}
        )

        # Express the pose in the data frame
        if self._yaw_deg != 0.0:
            if all(key in cartesian_state for key in obs_pose_keys):
                rotated = rotate_pose_z([cartesian_state[key] for key in obs_pose_keys], self._yaw_deg)
                cartesian_state.update(zip(obs_pose_keys, rotated.tolist(), strict=True))
            else:
                self._warn_throttled(
                    "data_frame_yaw", "data_frame_yaw_deg is set but the full pose is not observed; not rotating"
                )
        return cartesian_state

    def _read_camera(self, cam_key: str, cam) -> Any:
        try:
            return cam.async_read(timeout_ms=300)
        except TimeoutError:
            # Camera slower than the control loop: reuse the latest frame instead of failing
            return cam.read_latest(max_age_ms=1000)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Translate an action to ROS2Robot's keys, send it with ROS2Robot.send_action and return what was
        sent, in this robot's keys and data frame."""
        if self.config.offset_actions and OFFSET_ACTION_KEYS[0] in action:
            return self._send_offset_action(action)

        key_map = self._action_key_map()
        is_pose = self.config.action_type is ActionType.CARTESIAN_POSITION
        source_keys = list(key_map)
        if is_pose and source_keys[0] not in action:
            # `lerobot-rollout` returns the robot to its initial position by replaying the
            # observation keys (`current_robot_pose.N.pos`), so accept those as a pose target too.
            if self.config.zero_state:
                # ... but then they are zeros, not a pose. Raising makes lerobot-rollout skip the return.
                raise ValueError("Not sending observed poses as a target: zero_state reports them as 0")
            source_keys = self._obs_pose_keys

        values = [action[key] for key in source_keys]
        if is_pose:
            # Data frame -> robot frame
            values = rotate_pose_z(values, self._yaw_deg, inverse=True).tolist()
        sent = super().send_action(dict(zip(key_map.values(), values, strict=True)))

        sent_values = [sent[key] for key in key_map.values()]
        if is_pose:
            sent_values = rotate_pose_z(sent_values, self._yaw_deg).tolist()
        return dict(zip(key_map, sent_values, strict=True))

    def _send_offset_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send the measured pose moved by the action's offset, as an absolute pose with all its limits. Returns
        the offset actually applied, or the action unchanged if nothing was sent."""
        offset = np.array([action[key] for key in OFFSET_ACTION_KEYS], dtype=float)
        if not np.all(np.isfinite(offset)):
            logger.error(f"Dropping offset action with non-finite values: {offset}")
            return dict(action)
        # Like in training, the offset refers to the pose measured right when it is executed
        measured = self._measured_pose()
        target = None if measured is None else apply_pose_offset(measured, offset)
        if target is None:
            self._warn_throttled("offset_pose", "Dropping offset action: the measured pose is missing or invalid")
            return dict(action)
        sent = self.send_action(dict(zip(self._target_pose_keys, target.tolist(), strict=True)))
        applied = pose_offset(measured, [sent[key] for key in self._target_pose_keys])
        return dict(zip(OFFSET_ACTION_KEYS, (offset if applied is None else applied).tolist(), strict=True))


class SO101ROS(ROS2Robot):
    pass


class AnninAR4(ROS2Robot):
    pass

class UR10eSim(ROS2Robot):
    pass
 
class UR10eReal(ROS2Robot):
    pass


# lerobot finds the robot class by stripping "Config" from the config class name
class UR10eRealCrop(UR10eReal):
    pass


class UR10eRealNoJoints(UR10eReal):
    pass


class UR10eRealNoJointsCropped(UR10eReal):
    pass

class CartesianRobotReal(CartesianRobot):
    pass

class CartesianRobotSim(CartesianRobot):
    pass

class CartesianRobotSimNoMiddle(CartesianRobot):
    pass


class CartesianRobotRealNoMiddle(CartesianRobot):
    pass
