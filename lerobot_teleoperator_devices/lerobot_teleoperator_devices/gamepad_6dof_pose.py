import logging
import threading
import time
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from lerobot_robot_ros.ros_interface import POSE_6D_KEYS, POSE_QUAT_KEYS
from lerobot_robot_ros.safety import limit_position, limit_rotation
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from .config_gamepad_6dof_pose import Gamepad6DOFPoseTeleopConfig
from .gamepad_6dof import Gamepad6DOFTeleop

logger = logging.getLogger(__name__)

# Longer gaps between get_action() calls (e.g. a pause between episodes) do not move the target further
_MAX_DT_S = 0.1


class Gamepad6DOFPoseTeleop(Gamepad6DOFTeleop):
    """
    Gamepad teleop for ActionType.CARTESIAN_POSITION: outputs absolute end effector poses.

    Same mapping as Gamepad6DOFTeleop. The sticks move a target pose with up to `max_linear_velocity`
    / `max_angular_velocity`, in the robot's base_link like the velocity teleop. The target starts at
    the measured pose (read from `pose_topic`) and stays within `max_position_lead` /
    `max_rotation_lead` of it. With the sticks released, the robot stops at the last target.
    """

    config_class = Gamepad6DOFPoseTeleopConfig
    name = "gamepad_6dof_pose"

    def __init__(self, config: Gamepad6DOFPoseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._executor_thread: threading.Thread | None = None
        # Latest measured pose (position, orientation, time.monotonic() of arrival)
        self._measured: tuple[np.ndarray, R, float] | None = None
        self._target: tuple[np.ndarray, R] | None = None
        self._last_quat: np.ndarray | None = None
        self._last_time: float | None = None
        self._warned_stale = False

    @property
    def _pose_keys(self) -> list[str]:
        return POSE_6D_KEYS if self.config.use_6d_rotation else POSE_QUAT_KEYS

    @property
    def action_features(self) -> dict:
        names = self._pose_keys + (["gripper.pos"] if self.config.use_gripper else [])
        return {
            "dtype": "float32",
            "shape": (len(names),),
            "names": {name: i for i, name in enumerate(names)},
        }

    def connect(self) -> None:
        super().connect()
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("gamepad_6dof_pose_teleop")
        self._node.create_subscription(PoseStamped, self.config.pose_topic, self._pose_callback, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

        deadline = time.monotonic() + self.config.connect_timeout_s
        while self._measured is None:
            if time.monotonic() > deadline:
                self.disconnect()
                raise TimeoutError(
                    f"No end effector pose received on {self.config.pose_topic} within "
                    f"{self.config.connect_timeout_s:.0f} s. Is the robot / controller running?"
                )
            time.sleep(0.1)

    def _pose_callback(self, msg: PoseStamped) -> None:
        position = msg.pose.position
        orientation = msg.pose.orientation
        try:
            rotation = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        except ValueError:
            logger.error(f"Ignoring end effector pose with invalid orientation {orientation}")
            return
        self._measured = (np.array([position.x, position.y, position.z]), rotation, time.monotonic())

    def get_action(self) -> dict[str, Any]:
        if self.gamepad is None:
            raise RuntimeError("Gamepad is not connected. Please call connect() first.")

        self.gamepad.update()
        velocities = np.array(self.gamepad.get_6dof_deltas(), dtype=float)
        now = time.monotonic()
        dt = 0.0 if self._last_time is None else min(now - self._last_time, _MAX_DT_S)
        self._last_time = now
        self._update_target(velocities, dt, now)

        action_dict = dict(zip(self._pose_keys, self._target_pose(), strict=True))
        if self.config.use_gripper:
            action_dict["gripper.pos"] = self.gamepad.gripper_command()
        return action_dict

    def _update_target(self, velocities: np.ndarray, dt: float, now: float) -> None:
        """Move the target by the stick velocities, keeping it within the lead of the measured pose.

        The target is held while the measured pose is stale.
        """
        if self._measured is None:
            raise RuntimeError(f"No end effector pose received yet on {self.config.pose_topic}.")
        measured_position, measured_rot, stamp = self._measured
        if self._target is None:
            self._target = (measured_position, measured_rot)

        if now - stamp > self.config.max_pose_age_s:
            if not self._warned_stale:
                logger.warning(f"End effector pose is {now - stamp:.2f} s old, holding the target.")
                self._warned_stale = True
            return
        self._warned_stale = False

        position, rot = self._target
        position = position + velocities[:3] * self.config.max_linear_velocity * dt
        rot = R.from_rotvec(velocities[3:] * self.config.max_angular_velocity * dt) * rot
        position = limit_position(position, measured_position, max_step=self.config.max_position_lead)
        rot = limit_rotation(rot, measured_rot, self.config.max_rotation_lead)
        self._target = (position, rot)

    def _target_pose(self) -> list[float]:
        position, rot = self._target
        if self.config.use_6d_rotation:
            matrix = rot.as_matrix()
            return [*position, *matrix[:, 0], *matrix[:, 1]]
        quat = rot.as_quat()
        # q and -q are the same rotation: keep the sign of the previous action, so the actions stay continuous
        if self._last_quat is not None and np.dot(quat, self._last_quat) < 0:
            quat = -quat
        self._last_quat = quat
        return [*position, *quat]

    def disconnect(self) -> None:
        # Stop spinning before destroying the node the executor might still be using
        if self._executor:
            self._executor.shutdown(timeout_sec=5.0)
            self._executor = None
        if self._executor_thread:
            self._executor_thread.join(timeout=5.0)
            self._executor_thread = None
        if self._node:
            self._node.destroy_node()
            self._node = None
        self._measured = self._target = self._last_quat = self._last_time = None
        super().disconnect()
