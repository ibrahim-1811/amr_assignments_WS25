# YOUR CODE HERE

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

from math import atan2, sqrt, pi


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].
    """
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


class OdometryMotion(Node):
    """
    A ROS2 node that drives a robot toward a specified goal pose using odometry feedback.
    """

    def __init__(self):
        super().__init__('odometry_motion')
        
        # Declare and read parameters
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', -3.0)
        self.declare_parameter('goal_theta', -2.0)
        self.declare_parameter('threshold', 0.05)
        self.declare_parameter('linear_velocity', 1.0)
        self.declare_parameter('angular_velocity', 0.5)
        self.declare_parameter('publish_rate', 10.0)

        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.goal_theta = self.get_parameter('goal_theta').value
        self.threshold = self.get_parameter('threshold').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.angular_velocity = self.get_parameter('angular_velocity').value
        rate = self.get_parameter('publish_rate').value

        self.current_pose = None  

        # Subscribers & Publishers
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Timer for control loop
        timer_period = 1.0 / rate
        self.timer = self.create_timer(timer_period, self.control_loop)
        
        self.get_logger().info('OdometryMotion node initialized')

    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback to update the current robot pose from odometry.
        """
        self.current_pose = msg.pose.pose

    def control_loop(self) -> None:
        """
        Periodic control loop: compute and publish velocity commands.
        """
        if self.current_pose is None:
        
            return

        # Extract pose and orientation
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        quat = self.current_pose.orientation
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        
        dx = self.goal_x - x
        dy = self.goal_y - y
        distance_error = sqrt(dx**2 + dy**2)

        angle_to_goal = atan2(dy, dx)
        heading_error = normalize_angle(angle_to_goal - yaw)
        orientation_error = normalize_angle(self.goal_theta - yaw)

        cmd = Twist()

        # Decision logic
        if distance_error < self.threshold:
            # At goal position: adjust orientation only
            if abs(orientation_error) < self.threshold:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.get_logger().info('Goal reached')
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_velocity * (1 if orientation_error > 0 else -1)
        else:
            
            if abs(heading_error) > self.threshold:
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_velocity * (1 if heading_error > 0 else -1)
            else:
                cmd.linear.x = self.linear_velocity
                cmd.angular.z = 0.0

        # Publish command
        self.publisher.publish(cmd)

    def destroy_node(self) -> None:
        self.get_logger().info('Shutting down OdometryMotion node')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OdometryMotion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
    
# raise NotImplementedError()
