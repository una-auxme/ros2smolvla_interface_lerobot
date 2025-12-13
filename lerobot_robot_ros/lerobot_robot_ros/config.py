# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.robots import RobotConfig

from lerobot_roscam.roscam_config import ROS2CameraConfig


class ActionType(Enum):
    CARTESIAN_VELOCITY = "cartesian_velocity"
    CARTESIAN_VELOCITY_TWIST_MSG = "cartesian_velocity_twist_msg"
    JOINT_POSITION = "joint_position"
    JOINT_TRAJECTORY = "joint_trajectory"


class GripperActionType(Enum):
    TRAJECTORY = "trajectory"  # Use JointTrajectoryController for gripper
    ACTION = "action"  # Use GripperActionClient

class GripperType(Enum):
    PARALLEL_GRIPPER = "parallel_gripper"
    GRIPPER = "gripper"


@dataclass
class ROS2InterfaceConfig:
    # Namespace used by ros2_control / MoveIt2 nodes
    namespace: str = ""

    arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
    )
    gripper_joint_name: str = "gripper_joint"

    # Base link name for computing end effector pose / velocity
    # Only applicable for cartesian control
    base_link: str = "base_link"

    # Only applicable if velocity control is used.
    max_linear_velocity: float = 0.10
    max_angular_velocity: float = 0.25  # rad/s

    # Only applicable if position control is used.
    min_joint_positions: list[float] | None = None
    max_joint_positions: list[float] | None = None

    gripper_open_position: float = 0.0
    gripper_close_position: float = 1.0

    gripper_action_type: GripperActionType = GripperActionType.TRAJECTORY
    gripper_type: GripperType = GripperType.GRIPPER


@dataclass
class ROS2Config(RobotConfig):
    # Action type for controlling the robot. Can be 'cartesian_velocity' or 'joint_position'.
    action_type: ActionType = ActionType.JOINT_POSITION

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ROS2 interface configuration
    ros2_interface: ROS2InterfaceConfig = field(default_factory=ROS2InterfaceConfig)


@RobotConfig.register_subclass("annin_ar4_mk1")
@dataclass
class AnninAR4Config(ROS2Config):
    """Annin Robotics AR4 robot configuration - extends ROS2Config with
    AR4-specific settings
    """

    action_type: ActionType = ActionType.CARTESIAN_VELOCITY

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            gripper_joint_name="gripper_jaw1_joint",
            base_link="base_link",
            min_joint_positions=[-2.9671, -0.7330, -1.5533, -2.8798, -1.8326, -2.7053],
            max_joint_positions=[2.9671, 1.5708, 0.9076, 2.8798, 1.8326, 2.7053],
            gripper_open_position=0.014,
            gripper_close_position=0.0,
            gripper_action_type=GripperActionType.ACTION,
        ),
    )


@RobotConfig.register_subclass("so101_ros")
@dataclass
class SO101ROSConfig(ROS2Config):
    """Configuration for the ROS 2 version of SO101: https://github.com/Pavankv92/lerobot_ws."""

    action_type: ActionType = ActionType.JOINT_TRAJECTORY

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            arm_joint_names=["1", "2", "3", "4", "5"],
            gripper_joint_name="6",
            base_link="base",
            min_joint_positions=[-1.91986, -1.74533, -1.74533, -1.65806, -2.79253],
            max_joint_positions=[1.91986, 1.74533, 1.5708, 1.65806, 2.79253],
            gripper_open_position=1.74533,
            gripper_close_position=0.0,
        ),
    )

@RobotConfig.register_subclass("ur_10e_sim")
@dataclass
class UR10eSimConfig(ROS2Config):
    action_type: ActionType = ActionType.CARTESIAN_VELOCITY
    max_relative_target = 0.2
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "camera1": ROS2CameraConfig(
                topic='/top_cam/image',
                node_name="top_cam",
                camera_type="camera",
                rgb_encoding="passthrough",
                fps=30,
                width=1280,
                height=720,
            ),

            "camera2": ROS2CameraConfig(
                topic='/side_cam/image',
                node_name="side_cam",
                camera_type="camera",
                rgb_encoding="passthrough",
                fps=30,
                width=1280,
                height=720,
            ),

            "camera3": ROS2CameraConfig(
                topic='/world/default/model/ur/link/wrist_3_link/sensor/wrist_cam/image',
                node_name="wrist_cam",
                rgb_encoding="passthrough",
                fps=30,
                width=1280,
                height=720,
            ),
        }
    )

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            base_link="base_link",
            arm_joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            
            gripper_joint_name="robotiq_hande_left_finger_joint",
            gripper_action_type=GripperActionType.ACTION,
            gripper_type=GripperType.PARALLEL_GRIPPER,
            gripper_open_position=0.0249,
            gripper_close_position=0.0001,
            max_linear_velocity=0.5,  # m/s
            max_angular_velocity=0.25,  # rad/s
            min_joint_positions=[-4.71, -3.14, 0.0, -7.5, -3.14, -3.14],
            max_joint_positions=[1.57, 0.0, 5.0, 2.5, 0.0, 3.14],
        )
    )

@RobotConfig.register_subclass("ur_10e_real")
@dataclass
class UR10eRealConfig(ROS2Config):
    action_type: ActionType = ActionType.CARTESIAN_VELOCITY_TWIST_MSG
    max_relative_target = 0.2
    
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
             "camera1": ROS2CameraConfig(
                 topic='/top_camera/rgb/image_raw/compressed',
                 node_name="top_cam",
                 camera_type="camera",
                 rgb_encoding="rgb",
                 fps=30,
                 width=720,
                 height=1280,
                 rotation=Cv2Rotation.ROTATE_270
             ),

             "camera2": ROS2CameraConfig(
                 topic='/side_camera/rgb/image_raw/compressed',
                 node_name="side_cam",
                 camera_type="camera",
                 rgb_encoding="rgb",
                 fps=30,
                 width=1280,
                 height=720,
            ),

            "camera3": ROS2CameraConfig(
                topic='/wrist_camera/image_raw/compressed',
                node_name="wrist_cam",
                rgb_encoding="rgb",
                fps=30,
                width=1280,
                height=720,
            ),
        })

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            base_link="base_link",
            arm_joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            
            gripper_joint_name="robotiq_hande_left_finger_joint",
            gripper_action_type=GripperActionType.ACTION,
            gripper_type=GripperType.PARALLEL_GRIPPER,
            gripper_open_position=0.0249,
            gripper_close_position=0.0001,
            max_linear_velocity=0.05,  # m/s
            max_angular_velocity=0.25,  # rad/s
            min_joint_positions=[-4.71, -3.14, 0.0, -7.5, -3.14, -3.14],
            max_joint_positions=[1.57, 0.0, 5.0, 2.5, 0.0, 3.14],
        )
    )