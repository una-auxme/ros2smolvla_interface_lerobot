from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_gamepad_6dof import Gamepad6DOFTeleopConfig


@TeleoperatorConfig.register_subclass("gamepad_6dof_pose")
@dataclass
class Gamepad6DOFPoseTeleopConfig(Gamepad6DOFTeleopConfig):
    # Must match the robot's ros2_interface.use_6d_rotation, the actions use the robot's pose keys
    use_6d_rotation: bool = False

    # Measured end effector pose, in the frame the robot expects its pose commands in (its base_link)
    pose_topic: str = "/cartesian_motion_controller/current_pose"

    # Target speed at full stick deflection, in m/s and rad/s
    max_linear_velocity: float = 0.05
    max_angular_velocity: float = 0.3

    # Maximum distance (m) and rotation (rad) the target may lead the measured pose, so it does not run
    # away while the robot lags behind or is held back by its limits. Keep these below the robot's
    # max_cartesian_step / max_rotation_step, so the robot never shortens a command and the recorded
    # action is the pose actually sent.
    max_position_lead: float = 0.02
    max_rotation_lead: float = 0.1

    # Seconds connect() waits for the first pose
    connect_timeout_s: float = 15.0
    # The target is held while the measured pose is older than this many seconds
    max_pose_age_s: float = 0.5

    def __post_init__(self):
        for name in (
            "max_linear_velocity",
            "max_angular_velocity",
            "max_position_lead",
            "max_rotation_lead",
            "connect_timeout_s",
            "max_pose_age_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
