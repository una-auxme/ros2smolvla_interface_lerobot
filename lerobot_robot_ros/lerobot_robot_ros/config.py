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
from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.cameras.configs import ColorMode
from lerobot.robots import RobotConfig

from lerobot_camera_roscam.roscam_config import ROS2CameraConfig


class ActionType(Enum):
    CARTESIAN_VELOCITY = "cartesian_velocity"
    CARTESIAN_VELOCITY_TWIST_MSG = "cartesian_velocity_twist_msg"
    CARTESIAN_POSITION = "cartesian_position"
    JOINT_POSITION = "joint_position"
    JOINT_TRAJECTORY = "joint_trajectory"


class GripperActionType(Enum):
    TRAJECTORY = "trajectory"  # Use JointTrajectoryController for gripper
    ACTION = "action"  # Use GripperActionClient
    NONE = "none"

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

    # Seconds connect() waits for the first complete robot state and for the gripper action server
    connect_timeout_s: float = 15.0

    # Robot state (joint states, Cartesian pose) older than this many seconds is reported as stale
    max_state_age_s: float = 0.5

    # Observe the end effector orientation as the first two rotation-matrix columns ("pose.r11" ...
    # "pose.r32") instead of a quaternion ("pose.quat_x" ... "pose.quat_w"). ActionType.CARTESIAN_POSITION
    # actions use the same representation.
    use_6d_rotation: bool = False

    # Only used by CartesianRobot (the cartesian_robot_* configs), with either rotation representation.
    # Rotation about the z axis (degrees) that maps the robot's `base_link` frame into the data frame,
    # e.g. the frame a dataset was recorded or generated in: observations are rotated by +yaw before
    # they reach the policy, actions by -yaw before they are sent to the robot.
    data_frame_yaw_deg: float = 0.0

    # Only applicable for ActionType.CARTESIAN_POSITION. Maximum distance (m) a commanded target
    # may be away from the current pose. Targets further away are pulled back onto this sphere.
    max_cartesian_step: float | None = None
    # Only applicable for ActionType.CARTESIAN_POSITION. Maximum rotation (rad) a commanded target
    # may be away from the current orientation. Larger rotations are shortened along the same axis.
    max_rotation_step: float | None = None
    # Optional workspace box ((x_min, y_min, z_min), (x_max, y_max, z_max)) in `base_link`.
    # Either bound may be set on its own.
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None

    def __post_init__(self):
        for name in ("workspace_min", "workspace_max"):
            bound = getattr(self, name)
            if bound is not None and len(bound) != 3:
                raise ValueError(f"{name} must have 3 elements (x, y, z), got {bound}")
        if self.workspace_min is not None and self.workspace_max is not None:
            if any(lo > hi for lo, hi in zip(self.workspace_min, self.workspace_max, strict=True)):
                raise ValueError(f"workspace_min {self.workspace_min} exceeds workspace_max {self.workspace_max}")
        for name in ("max_cartesian_step", "max_rotation_step"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


@dataclass
class ROS2Config(RobotConfig):
    # Action type for controlling the robot. Can be 'cartesian_velocity' or 'joint_position'.
    action_type: ActionType = ActionType.JOINT_POSITION

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dict mapping each action key
    # (e.g. "shoulder_pan_joint.pos") to its own limit. For cartesian velocity actions it caps each normalized
    # velocity component instead.
    # Subclasses must override this with a type annotation, otherwise the dataclass silently keeps `None`.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    observation_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
    )

    # ROS2 interface configuration
    ros2_interface: ROS2InterfaceConfig = field(default_factory=ROS2InterfaceConfig)


_UR_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_UR_HANDE_JOINT_OBS = [f"{joint}.pos" for joint in _UR_ARM_JOINTS] + ["robotiq_hande_left_finger_joint.pos"]
_POSE_QUAT_OBS = ["pose.x", "pose.y", "pose.z", "pose.quat_x", "pose.quat_y", "pose.quat_z", "pose.quat_w"]
# (ymin, ymax, xmin, xmax) of the rotated 720 x 1280 top camera image, giving a 720 x 720 frame
_TOP_CAMERA_SQUARE_CROP = [560, 1280, 0, 720]


def _ur_hande_interface(max_linear_velocity: float) -> ROS2InterfaceConfig:
    """UR10e with a Robotiq Hand-E gripper."""
    return ROS2InterfaceConfig(
        base_link="base_link",
        arm_joint_names=list(_UR_ARM_JOINTS),
        gripper_joint_name="robotiq_hande_left_finger_joint",
        gripper_action_type=GripperActionType.ACTION,
        gripper_type=GripperType.PARALLEL_GRIPPER,
        gripper_open_position=0.0249,
        gripper_close_position=0.0001,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=0.25,
    )


def _sim_camera(topic: str, node_name: str, **kwargs) -> ROS2CameraConfig:
    return ROS2CameraConfig(
        topic=topic,
        node_name=node_name,
        camera_type="camera",
        # Workaround to leave sim image in RGB, since lerobot_camera_roscam assumes BGR input and
        # simply passes the image through
        color_mode=ColorMode.BGR,
        fps=30,
        **kwargs,
    )


def _real_camera(topic: str, node_name: str, width: int = 1280, height: int = 720, **kwargs) -> ROS2CameraConfig:
    return ROS2CameraConfig(
        topic=topic,
        node_name=node_name,
        camera_type="compressed_camera",
        color_mode=ColorMode.RGB,
        fps=30,
        width=width,
        height=height,
        **kwargs,
    )


def _real_cameras(crop_top: bool = False) -> dict[str, CameraConfig]:
    """Top (mounted rotated, optionally cropped to a square), side and wrist camera of the real cell."""
    if crop_top:
        top = _real_camera(
            "/top_camera/rgb/image_raw/compressed", "top_cam", width=720, height=720,
            rotation=Cv2Rotation.ROTATE_270, crop_box=list(_TOP_CAMERA_SQUARE_CROP),
        )
    else:
        top = _real_camera(
            "/top_camera/rgb/image_raw/compressed", "top_cam", width=720, height=1280,
            rotation=Cv2Rotation.ROTATE_270,
        )
    return {
        "camera1": top,
        "camera2": _real_camera("/side_camera/rgb/image_raw/compressed", "side_cam"),
        "camera3": _real_camera("/wrist_camera/image_raw/compressed", "wrist_cam"),
    }


@RobotConfig.register_subclass("ur_10e_sim")
@dataclass
class UR10eSimConfig(ROS2Config):
    action_type: ActionType = ActionType.CARTESIAN_VELOCITY_TWIST_MSG
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "camera1": _sim_camera("/top_cam/image", "top_cam", width=360, height=640, rotation=Cv2Rotation.ROTATE_270),
            "camera2": _sim_camera("/side_cam/image", "side_cam", width=640, height=360),
            "camera3": _sim_camera("/wrist_cam/image", "wrist_cam", width=640, height=360),
        }
    )
    observation_names: list[str] = field(default_factory=lambda: _UR_HANDE_JOINT_OBS + _POSE_QUAT_OBS)
    ros2_interface: ROS2InterfaceConfig = field(default_factory=lambda: _ur_hande_interface(max_linear_velocity=0.5))


@RobotConfig.register_subclass("ur_10e_real")
@dataclass
class UR10eRealConfig(ROS2Config):
    action_type: ActionType = ActionType.CARTESIAN_VELOCITY_TWIST_MSG
    cameras: dict[str, CameraConfig] = field(default_factory=_real_cameras)
    observation_names: list[str] = field(default_factory=lambda: _UR_HANDE_JOINT_OBS + _POSE_QUAT_OBS)
    ros2_interface: ROS2InterfaceConfig = field(default_factory=lambda: _ur_hande_interface(max_linear_velocity=0.05))


@RobotConfig.register_subclass("ur_10e_real_crop")
@dataclass
class UR10eRealCropConfig(UR10eRealConfig):
    """Top camera cropped to a square."""

    cameras: dict[str, CameraConfig] = field(default_factory=lambda: _real_cameras(crop_top=True))


@RobotConfig.register_subclass("ur_10e_real_no_joints")
@dataclass
class UR10eRealNoJointsConfig(UR10eRealConfig):
    """Only the end effector pose is observed."""

    observation_names: list[str] = field(default_factory=lambda: list(_POSE_QUAT_OBS))


@RobotConfig.register_subclass("ur_10e_real_no_joints_cropped")
@dataclass
class UR10eRealNoJointsCroppedConfig(UR10eRealConfig):
    """Only the end effector pose is observed, top camera cropped to a square."""

    cameras: dict[str, CameraConfig] = field(default_factory=lambda: _real_cameras(crop_top=True))
    observation_names: list[str] = field(default_factory=lambda: list(_POSE_QUAT_OBS))


@RobotConfig.register_subclass("cartesian_robot_real")
@dataclass
class CartesianRobotRealConfig(ROS2Config):
    action_type: ActionType = ActionType.CARTESIAN_POSITION
    
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
             "top_center_camera": ROS2CameraConfig(
                 topic='/top_camera/rgb/image_raw/compressed',
                 node_name="top_cam",
                 camera_type="compressed_camera",
                 color_mode=ColorMode.RGB,
                 fps=30,
                 width=720,
                 height=720,
                 rotation=Cv2Rotation.ROTATE_270,
                 crop_box=[560, 1280, 0, 720] # This crops the output to 720 x 720
             ),

             "top_left_camera": ROS2CameraConfig(
                 topic='/side_camera/rgb/image_raw/compressed',
                 node_name="side_cam",
                 camera_type="compressed_camera",
                 color_mode=ColorMode.RGB,
                 fps=30,
                 width=1280,
                 height=720,
            ),

            "middle_left_camera": ROS2CameraConfig(
                topic='/wrist_camera/image_raw/compressed',
                node_name="wrist_cam",
                camera_type="compressed_camera",
                color_mode=ColorMode.RGB,
                fps=30,
                width=1280,
                height=720,
            ),
        })

    observation_names: list[str] = field(
        default_factory=lambda: [
        "current_robot_pose.0.pos",
        "current_robot_pose.1.pos",
        "current_robot_pose.2.pos",
        "current_robot_pose.3.pos",
        "current_robot_pose.4.pos",
        "current_robot_pose.5.pos",
        "current_robot_pose.6.pos",
        "current_robot_pose.7.pos",
        "current_robot_pose.8.pos",
        ]
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
            use_6d_rotation=True,

            gripper_action_type=GripperActionType.NONE,
        )
    )

@RobotConfig.register_subclass("cartesian_robot_sim")
@dataclass
class CartesianRobotSimConfig(ROS2Config):
    action_type: ActionType = ActionType.CARTESIAN_POSITION
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "top_center_camera": ROS2CameraConfig(
                topic='/top_cam/image',
                node_name="top_cam",
                camera_type="camera",
                color_mode=ColorMode.BGR, # Workaround to leave sim image in RGB, since lerobot_camera_roscam assumes BGR input and simply passes the image through
                fps=30,
                width=360,
                height=640,
                rotation=Cv2Rotation.ROTATE_270
            ),

            "top_left_camera": ROS2CameraConfig(
                topic='/side_cam/image',
                node_name="side_cam",
                camera_type="camera",
                color_mode=ColorMode.BGR, # Workaround to leave sim image in RGB, since lerobot_camera_roscam assumes BGR input and simply passes the image through
                fps=30,
                width=640,
                height=360,
            ),

            "middle_left_camera": ROS2CameraConfig(
                topic='/wrist_cam/image',
                node_name="wrist_cam",
                camera_type="camera",
                color_mode=ColorMode.BGR, # Workaround to leave sim image in RGB, since lerobot_camera_roscam assumes BGR input and simply passes the image through
                fps=30,
                width=640,
                height=360,
            ),
        }
    )

    observation_names: list[str] = field(
        default_factory=lambda: [
        "current_robot_pose.0.pos",
        "current_robot_pose.1.pos",
        "current_robot_pose.2.pos",
        "current_robot_pose.3.pos",
        "current_robot_pose.4.pos",
        "current_robot_pose.5.pos",
        "current_robot_pose.6.pos",
        "current_robot_pose.7.pos",
        "current_robot_pose.8.pos",
        ]
    )

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            base_link="world",
            arm_joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            use_6d_rotation=True,
            data_frame_yaw_deg=180.0,

            gripper_action_type=GripperActionType.NONE,
        )
    )


_POSE_OBS_NAMES = [f"current_robot_pose.{n}.pos" for n in range(9)]


@RobotConfig.register_subclass("cartesian_robot_sim_no_middle")
@dataclass
class CartesianRobotSimNoMiddleConfig(ROS2Config):
    """Sim, without the middle_left camera. Images are 640x360 (W x H), unrotated, like the synthetic training data."""

    action_type: ActionType = ActionType.CARTESIAN_POSITION
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "top_center_camera": ROS2CameraConfig(
                topic="/top_cam/image",
                node_name="top_cam",
                camera_type="camera",
                color_mode=ColorMode.BGR,  # see workaround note in the other sim configs
                fps=30,
                width=640,
                height=360,
            ),
            "top_left_camera": ROS2CameraConfig(
                topic="/side_cam/image",
                node_name="side_cam",
                camera_type="camera",
                color_mode=ColorMode.BGR,  # see workaround note in the other sim configs
                fps=30,
                width=640,
                height=360,
            ),
        }
    )
    observation_names: list[str] = field(default_factory=lambda: list(_POSE_OBS_NAMES))
    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            base_link="world",
            arm_joint_names=list(_UR_ARM_JOINTS),
            use_6d_rotation=True,
            data_frame_yaw_deg=180.0,
            gripper_action_type=GripperActionType.NONE,
            max_cartesian_step=0.05,
        )
    )


@RobotConfig.register_subclass("cartesian_robot_real_no_middle")
@dataclass
class CartesianRobotRealNoMiddleConfig(ROS2Config):
    """Real robot, without the middle_left camera. 1280x720 feeds are downscaled to 640x360, unrotated and uncropped."""

    action_type: ActionType = ActionType.CARTESIAN_POSITION
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "top_center_camera": ROS2CameraConfig(
                topic="/side_camera/rgb/image_raw/compressed",
                node_name="side_cam",
                camera_type="compressed_camera",
                color_mode=ColorMode.RGB,
                fps=30,
                width=640,
                height=360,
                resize=(640, 360),
            ),
            "top_left_camera": ROS2CameraConfig(
                topic="/top_camera/rgb/image_raw/compressed",
                node_name="top_cam",
                camera_type="compressed_camera",
                color_mode=ColorMode.RGB,
                fps=30,
                width=640,
                height=360,
                resize=(640, 360),
            ),
        }
    )
    observation_names: list[str] = field(default_factory=lambda: list(_POSE_OBS_NAMES))
    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            base_link="world",
            arm_joint_names=list(_UR_ARM_JOINTS),
            data_frame_yaw_deg=180.0,
            use_6d_rotation=True,
            gripper_action_type=GripperActionType.NONE,
            max_cartesian_step=0.02,
            workspace_min=(-0.683, -0.349, 0.255),
            workspace_max=(-0.253, 0.350, 0.570),
        )
    )
