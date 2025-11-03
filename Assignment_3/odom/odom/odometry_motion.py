import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import atan2
from tf_transformations import euler_from_quaternion

class OdometryMotion(Node):

    def __init__(self):
        super().__init__('odometry_motion')
        self.subscriber_ = self.create_subscription(Odometry, '/odom', self.callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.goal = {
            "x":2.0,
            "y":-3.0,
            "theta":-2.0
        }

        self.treshold = 0.05

        self.velocity = 1.
        self.angularVelocity = 0.5


    def callback(self, msg: Odometry):
        command = Twist()

        robotPose = msg.pose.pose
        eulerAngle = euler_from_quaternion([robotPose.orientation.x, robotPose.orientation.y, robotPose.orientation.z, robotPose.orientation.w])

        positionError = ( ( robotPose.position.x - self.goal["x"] )**2 + (robotPose.position.y - self.goal["y"])**2 )**0.5
        orientationError = self.goal["theta"] - eulerAngle[2]
        faceGoalAngle = atan2(self.goal["y"] - robotPose.position.y, self.goal["x"] - robotPose.position.x)
        faceGoalError = faceGoalAngle - eulerAngle[2]

        if positionError < self.treshold and abs(orientationError) < self.treshold:
            # goal reached 
            command.linear.x = 0.
            command.angular.z = 0.

        elif positionError < self.treshold:
            # alignment
            command.angular.z = self.angularVelocity if orientationError > 0 else -self.angularVelocity
            command.linear.x = 0.
            
        elif abs(faceGoalError) < self.treshold:
            # reach goal position 
            command.angular.z = 0.
            command.linear.x = self.velocity
        
        else:
            # face the destination
            command.angular.z = self.angularVelocity if faceGoalError > 0 else -self.angularVelocity
            command.linear.x = 0.

        self.publisher_.publish(command)


def main(args=None):
    rclpy.init(args=args)

    odm = OdometryMotion()

    rclpy.spin(odm)

    odm.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()