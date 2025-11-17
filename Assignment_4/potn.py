import rclpy
from rclpy.node import Node


from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from tf_transformations import euler_from_quaternion
import tf2_ros 
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import do_transform_pose

from math import sin, cos, isinf, atan2
import numpy as np

class PotentialField(Node):

    def __init__(self):
        super().__init__('potential_field')
        self.subscriber1_ = self.create_subscription(LaserScan, '/scan', self.callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.goal = {
            "x":4.0,
            "y":10.0,
            "theta":-1.0
        }

        self.psGoal = PoseStamped()
        self.psGoal.header.frame_id = 'odom'
        self.psGoal.pose.position.x = self.goal["x"]
        self.psGoal.pose.position.y = self.goal["y"]
        self.psGoal.pose.orientation.w = cos(self.goal["theta"] * 0.5)
        self.psGoal.pose.orientation.z = sin(self.goal["theta"] * 0.5)

        self.ka = 1
        self.kr = 0.5

        self.maxVelocity = 3

        self.repulsiveThreshold = 20 
        self.errorThreshold = 0.1

        self.angularVelocity = 0.8

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timeWD = self.get_clock().now()
        self.goalDistanceWD = 0
        
    def getAttractive(self, g_b):
        return self.ka*g_b/np.linalg.norm(g_b)

    def getRepulsive(self, obstacles):
        vr = 0

        for o in obstacles:
            norm = np.linalg.norm(o)
            if norm < self.repulsiveThreshold:
                vr -= self.kr * (1/norm - 1/self.repulsiveThreshold )*(1/norm**2)*o/norm
        
        return vr
    
    def polar2cart(self, polar):
        cartesian_data = list()
        for ro,theta in polar:
            if not isinf(ro):
                x = ro * cos(theta)
                y = ro * sin(theta)
                cartesian_data.append(np.array([x, y]))
    
        return np.array(cartesian_data)
    
    def isRobotStuck(self, goalDistance):

        if abs(goalDistance - self.goalDistanceWD) > 1:
            self.timeWD = self.get_clock().now()
            self.goalDistanceWD = goalDistance
        
        isStuck = self.get_clock().now().seconds_nanoseconds()[0] - self.timeWD.seconds_nanoseconds()[0] > 5.

        if isStuck: 
            self.timeWD = self.get_clock().now()

        return isStuck
        
    def goToGoal(self, msg, position)->Twist:
        command = Twist()

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        polar = zip(msg.ranges, angles)

        cartesian = self.polar2cart(polar) # all the obstacles

        transformed_point = np.array([position.x, position.y])
        force = None
        
        if self.isRobotStuck(np.linalg.norm(transformed_point)):
            force = np.random.random(2)*5 
        else:        
            force = self.getAttractive(transformed_point) + self.getRepulsive(cartesian)

        if np.linalg.norm(force) > self.maxVelocity:
            force = force/np.linalg.norm(force) * self.maxVelocity # Upper bound the maximum velocity 

        command.linear.x = force[0]
        command.linear.y = force[1]

        command.angular.z = np.clip(atan2(command.linear.y, command.linear.x), -self.angularVelocity , self.angularVelocity)

        return command
    
    def alignToAngle(self, angle):
        command = Twist()
        command.linear.x = 0.
        command.linear.y = 0.
        command.angular.z = self.angularVelocity if angle> 0 else -self.angularVelocity
        return command
    
    def callback(self, msg: LaserScan):
        
        command = Twist()

        if self.tf_buffer.can_transform("base_link", "odom", rclpy.time.Time().to_msg()):
            
            transform_stamped = self.tf_buffer.lookup_transform("base_link", "odom", rclpy.time.Time().to_msg())
            transformed_pose = do_transform_pose(self.psGoal.pose, transform_stamped)
            eulerAngle = euler_from_quaternion([transformed_pose.orientation.x,transformed_pose.orientation.y,transformed_pose.orientation.z,transformed_pose.orientation.w])[2]

            positionError = (transformed_pose.position.x**2 + transformed_pose.position.y**2)**0.5

            if positionError > self.errorThreshold:
                command = self.goToGoal(msg, transformed_pose.position)

            elif abs(eulerAngle) > self.errorThreshold:
                command = self.alignToAngle(eulerAngle)
            else:
                command.linear.x = 0.; command.linear.y = 0.; command.angular.z = 0.

        self.publisher_.publish(command)


def main(args=None):
    rclpy.init(args=args)

    odm = PotentialField()

    rclpy.spin(odm)

    odm.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
