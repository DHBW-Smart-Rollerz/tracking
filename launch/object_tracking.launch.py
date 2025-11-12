"""Launch file for the Object Tracking Node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    """Generate the launch description for the object tracking node."""
    # Get the package directory
    package_dir = get_package_share_directory("tracking")

    # Path to the config file
    config_file = os.path.join(package_dir, "config", "tracking_params.yaml")

    # Create the tracking node
    tracking_node = Node(
        package="tracking",
        executable="object_tracking_node",
        name="object_tracking_node",
        output="screen",
        parameters=[config_file],
        emulate_tty=True,
    )

    return LaunchDescription([tracking_node])
