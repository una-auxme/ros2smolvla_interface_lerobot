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
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from control_msgs.action import ParallelGripperCommand, GripperCommand
from lerobot.utils.errors import DeviceNotConnectedError
from rclpy import qos
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import Executor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TwistStamped, PoseStamped

from .config import ActionType, GripperActionType, GripperType, ROS2InterfaceConfig
from .moveit_servo import MoveIt2Servo

logger = logging.getLogger(__name__)


def rotate_pose9_z(pose, yaw_deg: float, inverse: bool = False) -> np.ndarray:
    """Rotate a 9-D pose (xyz + first two rotation-matrix columns) about the z axis.

    Applies p' = Rz(yaw) p to the position and to both rotation columns, i.e. R' = Rz(yaw) R.
    With `inverse=True` the opposite rotation is applied.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (9,):
        raise ValueError(f"Expected a 9-D pose, got shape {pose.shape}")
    if yaw_deg == 0.0:
        return pose.copy()
    yaw = np.deg2rad(-yaw_deg if inverse else yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.concatenate([rz @ pose[0:3], rz @ pose[3:6], rz @ pose[6:9]])



class ROS2Interface:
    """Class to interface with a MoveIt2 manipulator.

    This class supports both JointGroupPositionController and JointTrajectoryController
    from ros2_control for arm control, depending on the configuration:

    - ActionType.JOINT_POSITION:
      Uses JointGroupPositionController.
      Publishes Float64MultiArray messages to '/position_controller/commands'

    - ActionType.JOINT_TRAJECTORY:
      Uses JointTrajectoryController.
      Publishes JointTrajectory messages to '/arm_controller/joint_trajectory'

    The gripper control also supports both trajectory and action-based control
    via the gripper_action_type configuration option.
    """

    def __init__(self, config: ROS2InterfaceConfig, action_type: ActionType):
        self.config = config
        self.action_type = action_type
        self.robot_node: Node | None = None
        self.pos_cmd_pub: Publisher | None = None
        self.traj_cmd_pub: Publisher | None = None
        self.twist_pub: Publisher | None = None
        self.joint_state_sub: Subscription | None = None
        self.cart_pose_sub: Subscription | None = None 
        self.gripper_action_client: ActionClient | None = None
        self.gripper_traj_pub: Publisher | None = None
        self.executor: Executor | None = None
        self.moveit2_servo: MoveIt2Servo | None = None
        self.executor_thread: threading.Thread | None = None
        self.is_connected = False
        self._last_joint_state: dict[str, dict[str, float]] | None = None

    def connect(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self.robot_node = Node("moveit2_interface_node", namespace=self.config.namespace)
        if self.action_type == ActionType.JOINT_POSITION:
            self.pos_cmd_pub = self.robot_node.create_publisher(
                Float64MultiArray, "/position_controller/commands", 10
            )
        elif self.action_type == ActionType.JOINT_TRAJECTORY:
            self.traj_cmd_pub = self.robot_node.create_publisher(
                JointTrajectory, "/arm_controller/joint_trajectory", 10
            )
        elif self.action_type == ActionType.CARTESIAN_VELOCITY:
            self.moveit2_servo = MoveIt2Servo(
                node=self.robot_node,
                frame_id=self.config.base_link,
                callback_group=ReentrantCallbackGroup(),
            )
        elif self.action_type == ActionType.CARTESIAN_VELOCITY_TWIST_MSG:
            self.twist_pub = self.robot_node.create_publisher(
             TwistStamped,
             "/servo_node/delta_twist_cmds",
             qos.QoSProfile(
                 durability=qos.QoSDurabilityPolicy.VOLATILE,
                 reliability=qos.QoSReliabilityPolicy.RELIABLE,
                 history=qos.QoSHistoryPolicy.KEEP_ALL
             ))
            self.cart_pose_sub = self.robot_node.create_subscription(
                PoseStamped,
                "/cartesian_motion_controller/current_pose",
                self._cart_pose_callback,
                10
            )
            self._twist_msg = TwistStamped()

        elif self.action_type == ActionType.CARTESIAN_POSITION:
            self.pos_cmd_pub = self.robot_node.create_publisher(
                PoseStamped,
                "/cartesian_motion_controller/target_frame",
                qos.QoSProfile(
                    durability=qos.QoSDurabilityPolicy.VOLATILE,
                    reliability=qos.QoSReliabilityPolicy.RELIABLE,
                    history=qos.QoSHistoryPolicy.KEEP_ALL
                ))
            self.cart_pose_sub = self.robot_node.create_subscription(
                PoseStamped,
                "/cartesian_motion_controller/current_pose",
                self._cart_pose_callback,
                10
            )

        if self.config.gripper_action_type == GripperActionType.TRAJECTORY:
            self.gripper_traj_pub = self.robot_node.create_publisher(
                JointTrajectory, "/gripper_controller/joint_trajectory", 10
            )
        elif self.config.gripper_action_type == GripperActionType.ACTION:
            if self.config.gripper_type == GripperType.GRIPPER:
                self.gripper_action_client = ActionClient(
                    self.robot_node,
                    GripperCommand,
                    "/gripper_controller/gripper_cmd",
                    callback_group=ReentrantCallbackGroup(),
                )
                self._goal_msg = GripperCommand.Goal()
            elif self.config.gripper_type == GripperType.PARALLEL_GRIPPER:
                self.gripper_action_client = ActionClient(
                    self.robot_node,
                    ParallelGripperCommand,
                    "/gripper_action_controller/gripper_cmd",
                    callback_group=ReentrantCallbackGroup(),
                )
                self.gripper_action_client.wait_for_server()    
                self._goal_msg = ParallelGripperCommand.Goal()
            else:
                self.robot_node.get_logger().warn("No GripperType defined, how did this happen?")
        else:
            pass

        self.joint_state_sub = self.robot_node.create_subscription(
            JointState,
            "joint_states",
            self._joint_state_callback,
            10,
        )

        # Create and start the executor in a separate thread
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.robot_node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        time.sleep(1)  # Give some time to connect to services

        # Block until the first joint state (and Cartesian pose, if used) arrived, so that the very
        # first get_observation() right after connect() does not fail.
        needs_pose = self.cart_pose_sub is not None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            state = self._last_joint_state or {}
            if "position" in state and (not needs_pose or "cart_pose" in state):
                break
            time.sleep(0.1)
        else:
            logger.warning(
                "No joint state%s received within 15 s. Is the robot / controller running?",
                " / Cartesian pose" if needs_pose else "",
            )

        self.is_connected = True

    def send_joint_position_command(self, joint_positions: list[float], unnormalize: bool = True) -> None:
        """
        Send a command to the robot's joints.
        Args:
            joint_positions (list[float]): The target positions for the joints.
            unnormalize (bool): Whether to unnormalize the joint positions based on the robot's configuration.
        """
        if not self.robot_node:
            raise DeviceNotConnectedError("ROS2Interface is not connected. You need to call `connect()`.")

        if unnormalize:
            if self.config.min_joint_positions is None or self.config.max_joint_positions is None:
                raise ValueError(
                    "Joint position normalization requires min and max joint positions to be set."
                )
            joint_positions = [
                min(max(pos, min_pos), max_pos)
                for pos, min_pos, max_pos in zip(
                    joint_positions,
                    self.config.min_joint_positions,
                    self.config.max_joint_positions,
                    strict=True,
                )
            ]

        if len(joint_positions) != len(self.config.arm_joint_names):
            raise ValueError(
                f"Expected {len(self.config.arm_joint_names)} joint positions, but got {len(joint_positions)}."
            )

        if self.action_type == ActionType.JOINT_TRAJECTORY:
            if self.traj_cmd_pub is None:
                raise DeviceNotConnectedError("Trajectory command publisher is not initialized.")
            msg = JointTrajectory()
            msg.joint_names = self.config.arm_joint_names
            point = JointTrajectoryPoint()
            point.positions = joint_positions
            msg.points = [point]
            self.traj_cmd_pub.publish(msg)
        else:
            if self.pos_cmd_pub is None:
                raise DeviceNotConnectedError("Position command publisher is not initialized.")
            msg = Float64MultiArray()
            msg.data = joint_positions
            self.pos_cmd_pub.publish(msg)

    def servo(self, linear, angular, normalize: bool = True) -> None:
        if normalize:
            linear = [v * self.config.max_linear_velocity for v in linear]
            angular = [v * self.config.max_angular_velocity for v in angular]

        if self.action_type == ActionType.CARTESIAN_VELOCITY:
            if not self.moveit2_servo:
                raise DeviceNotConnectedError("ROS2Interface is not connected. You need to call `connect()`.")

            self.moveit2_servo.servo(linear=linear, angular=angular)
            return
        
        elif self.action_type == ActionType.CARTESIAN_VELOCITY_TWIST_MSG:
            twist_msg = TwistStamped()
            twist_msg.header.frame_id = self.config.base_link
            twist_msg.header.stamp = self.robot_node.get_clock().now().to_msg()
            twist_msg.twist.linear.x = float(linear[0])
            twist_msg.twist.linear.y = float(linear[1])
            twist_msg.twist.linear.z = float(linear[2])
            twist_msg.twist.angular.x = float(angular[0])
            twist_msg.twist.angular.y = float(angular[1])
            twist_msg.twist.angular.z = float(angular[2])
            self.twist_pub.publish(twist_msg)
            return

    def send_cartesian_position_command(self, pose) -> bool:
        """Send an absolute pose (xyz + first two rotation-matrix columns) to the motion controller.

        Returns False (and sends nothing) if the pose is invalid.
        """
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (9,) or not np.all(np.isfinite(pose)):
            logger.error(f"Dropping invalid pose command: {pose}")
            return False

        position = pose[:3].copy()

        # Split the 6D representation into two 3D vectors
        v1 = pose[3:6]
        v2 = pose[6:9]
        if np.linalg.norm(v1) < 1e-6:
            logger.error("Dropping pose command: first rotation column has ~zero norm")
            return False
        u2_check = v2 - np.dot(v1 / np.linalg.norm(v1), v2) * (v1 / np.linalg.norm(v1))
        if np.linalg.norm(u2_check) < 1e-6:
            logger.error("Dropping pose command: rotation columns are parallel")
            return False

        # Safety: keep target inside the workspace box and within max_cartesian_step of the current pose
        if self.config.workspace_min is not None and self.config.workspace_max is not None:
            position = np.clip(position, self.config.workspace_min, self.config.workspace_max)
        if self.config.max_cartesian_step is not None:
            current = (self._last_joint_state or {}).get("cart_pose")
            if current is None:
                logger.error("Dropping pose command: current pose unknown, cannot apply step limit")
                return False
            cur_pos = np.array([current["pose.x"], current["pose.y"], current["pose.z"]])
            step = position - cur_pos
            dist = np.linalg.norm(step)
            if dist > self.config.max_cartesian_step:
                position = cur_pos + step * (self.config.max_cartesian_step / dist)

        # 1. Gram-Schmidt Orthogonalization
        # Normalize the first vector (this forms the first column of the matrix)
        c1 = v1 / np.linalg.norm(v1)
        
        # Make the second vector orthogonal to the first, then normalize (second column)
        u2 = v2 - np.dot(c1, v2) * c1
        c2 = u2 / np.linalg.norm(u2)
        
        # The third column is the cross product of the first two orthogonal vectors
        c3 = np.cross(c1, c2)
        
        # 2. Reconstruct the 3x3 Rotation Matrix (stacking columns)
        rot_matrix = np.column_stack((c1, c2, c3))
        
        # 3. Convert to target format (XYZW Quaternion)
        quat_xyzw = R.from_matrix(rot_matrix).as_quat()

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.config.base_link
        pose_msg.header.stamp = self.robot_node.get_clock().now().to_msg()

        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])

        pose_msg.pose.orientation.x = float(quat_xyzw[0])
        pose_msg.pose.orientation.y = float(quat_xyzw[1])
        pose_msg.pose.orientation.z = float(quat_xyzw[2])
        pose_msg.pose.orientation.w = float(quat_xyzw[3])

        self.pos_cmd_pub.publish(pose_msg)
        return True

    def send_gripper_command(self, position: float, unnormalize: bool = False) -> bool:
        """
        Send a command to the gripper to move to a specific position.
        Args:
            position (float): The target position for the gripper (0=open, 1=closed).
        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        if not self.robot_node:
            raise DeviceNotConnectedError("ROS2Interface is not connected. You need to call `connect()`.")

        if unnormalize:
            # Map normalized position (0=open, 1=closed) to actual gripper joint position
            open_pos = self.config.gripper_open_position
            closed_pos = self.config.gripper_close_position
            gripper_goal = open_pos + position * (closed_pos - open_pos)
        else:
            gripper_goal = position

        if self.config.gripper_action_type == GripperActionType.TRAJECTORY:
            if self.gripper_traj_pub is None:
                raise DeviceNotConnectedError("Gripper command publisher is not initialized.")
            msg = JointTrajectory()
            msg.joint_names = [self.config.gripper_joint_name]
            point = JointTrajectoryPoint()
            point.positions = [float(gripper_goal)]
            msg.points = [point]
            self.gripper_traj_pub.publish(msg)
            return True
        else:
            if not self.gripper_action_client:
                raise DeviceNotConnectedError("Gripper action client is not initialized.")

            if not self.gripper_action_client.wait_for_server(timeout_sec=1.0):
                logger.error("Gripper action server not available")
                return False

            if self.config.gripper_type == GripperType.GRIPPER:
                self._goal_msg.command.position = float(gripper_goal)
            elif self.config.gripper_type == GripperType.PARALLEL_GRIPPER:
                self._goal_msg.command.position = [float(gripper_goal)]

            if not (resp := self.gripper_action_client.send_goal_async(self._goal_msg)):
                logger.error("Failed to send gripper command")
                return False
            return True

    @property
    def joint_state(self) -> dict[str, dict[str, float]] | None:
        """Get the last received joint state."""
        return self._last_joint_state

    def _joint_state_callback(self, msg: "JointState") -> None:
        self._last_joint_state = self._last_joint_state or {}
        positions = {}
        velocities = {}
        name_to_index = {name: i for i, name in enumerate(msg.name)}

        for joint_name in self.config.arm_joint_names:
            idx = name_to_index.get(joint_name)
            if idx is None:
                raise ValueError(f"Joint '{joint_name}' not found in joint state.")
            positions[(joint_name + ".pos")] = msg.position[idx]
            velocities[(joint_name + ".pos")] = msg.velocity[idx]

        if self.config.gripper_action_type is not GripperActionType.NONE:
            idx = name_to_index.get(self.config.gripper_joint_name)
            if idx is None:
                raise ValueError(
                    f"Gripper joint '{self.config.gripper_joint_name}' not found in joint state."
                )
            positions[f"{self.config.gripper_joint_name}.pos"] = msg.position[idx]
            velocities[f"{self.config.gripper_joint_name}.pos"] = msg.velocity[idx]

        self._last_joint_state["position"] = positions
        self._last_joint_state["velocity"] = velocities

    def _cart_pose_callback(self, msg: "PoseStamped") -> None:
        """
        Callback function for the PoseStamped subscriber.
        Extracts Cartesian position and converts orientation based on config.
        """
        self._last_joint_state = self._last_joint_state or {}
        
        position = msg.pose.position
        orientation = msg.pose.orientation

        cart_pose = {
            "pose.x": position.x,
            "pose.y": position.y,
            "pose.z": position.z,
        }

        # Toggle via your config (e.g., self.config.use_6d_rotation or an Enum)
        if getattr(self.config, 'use_6d_rotation', False): 
            # Convert XYZW quaternion to 3x3 matrix
            r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
            rot_mat = r.as_matrix()
            
            # Extract first two columns for 6D representation
            cart_pose.update({
                "pose.r11": float(rot_mat[0, 0]),
                "pose.r21": float(rot_mat[1, 0]),
                "pose.r31": float(rot_mat[2, 0]),
                "pose.r12": float(rot_mat[0, 1]),
                "pose.r22": float(rot_mat[1, 1]),
                "pose.r32": float(rot_mat[2, 1]),
            })
        else:
            # Fallback to standard quaternion
            cart_pose.update({
                "pose.quat_x": orientation.x,
                "pose.quat_y": orientation.y,
                "pose.quat_z": orientation.z,
                "pose.quat_w": orientation.w,
            })

        self._last_joint_state["cart_pose"] = cart_pose
    def disconnect(self):
        if self.joint_state_sub:
            self.joint_state_sub.destroy()
            self.joint_state_sub = None
        if self.pos_cmd_pub:
            self.pos_cmd_pub.destroy()
            self.pos_cmd_pub = None
        if self.traj_cmd_pub:
            self.traj_cmd_pub.destroy()
            self.traj_cmd_pub = None
        if self.twist_pub:
            self.twist_pub.destroy()
            self.twist_pub = None
        if self.gripper_action_client:
            self.gripper_action_client.destroy()
            self.gripper_action_client = None
        if self.gripper_traj_pub:
            self.gripper_traj_pub.destroy()
            self.gripper_traj_pub = None
        if self.robot_node:
            self.robot_node.destroy_node()
            self.robot_node = None
        if self.moveit2_servo:
            self.moveit2_servo = None

        if self.executor:
            self.executor.shutdown()
            self.executor = None
        if self.executor_thread:
            self.executor_thread.join()
            self.executor_thread = None

        self.is_connected = False
