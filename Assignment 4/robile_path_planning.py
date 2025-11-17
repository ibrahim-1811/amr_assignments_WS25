import rclpy
from rclpy.node import Node
import numpy as np

# ROS 2 message types
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import euler_from_quaternion

class PotentialFieldPlanner(Node):
    """
    A simple potential field planner that drives the robot to a goal while avoiding obstacles.
    Uses force-based movement even during final orientation to prevent getting stuck.
    """
    
    def __init__(self):
        super().__init__('potential_field_planner')
        
        # ========== SETTINGS YOU CAN TWEAK ==========
        self.setup_parameters()
        self.setup_goal()
        self.setup_ros_communication()
        
        self.get_logger().info('🚀 Potential Field Planner Ready!')
        self.get_logger().info(f'🎯 Goal: ({self.goal_x}, {self.goal_y}) with angle: {self.goal_th}')

    def setup_parameters(self):
        """Set all the tuning parameters for the planner"""
        # How strongly the goal pulls the robot (like magnet strength)
        self.attraction_strength = 0.5    # ka
        
        # How strongly obstacles push the robot away  
        self.repulsion_strength = 2.0     # kr
        
        # How far away obstacles can affect the robot (meters)
        self.obstacle_influence_distance = 1.5  # rho0
        
        # Maximum speeds
        self.max_forward_speed = 0.5
        self.max_turn_speed = 1.0
        
        # How quickly the robot turns toward its desired direction
        self.turn_responsiveness = 2.0    # angular_gain

    def setup_goal(self):
        """Define where we want the robot to go"""
        # Goal position (x, y) in meters
        self.goal_x = 4.0
        self.goal_y = 10.0
        self.goal_th = -1.0  # Final orientation in radians
        
        # How close we need to get to consider goal reached
        self.position_close_enough = 0.2      # meters
        self.angle_close_enough = 0.05        # radians
        
        # Track if we're in final adjustment phase
        self.final_adjustment_mode = False
        self.stuck_counter = 0
        self.last_angle_error = None

    def setup_ros_communication(self):
        """Set up ROS topics and listeners"""
        # For listening to robot position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # For sending movement commands
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # For reading laser scanner data
        self.create_subscription(LaserScan, 'scan', self.process_laser_scan, 1)

    def process_laser_scan(self, laser_data):
        """
        Main function called when new laser data arrives.
        Calculates where to move based on goal attraction and obstacle repulsion.
        """
        # Step 1: Find where the robot is
        robot_position, robot_angle = self.get_robot_pose()
        if robot_position is None:
            return  # Couldn't get position
            
        # Step 2: Check if we reached the goal position
        distance_to_goal = self.calculate_distance(robot_position)
        
        if distance_to_goal < self.position_close_enough:
            # We're close to goal - enter final adjustment phase
            if not self.final_adjustment_mode:
                self.get_logger().info('🎯 Position reached! Starting final adjustment...')
                self.final_adjustment_mode = True
            
            # Use combined forces for final adjustment (prevents getting stuck)
            movement_command = self.final_orientation_adjustment(robot_position, robot_angle, laser_data)
        else:
            # Normal navigation to goal
            movement_command = self.normal_navigation(robot_position, robot_angle, laser_data)
        
        # Send the movement command
        self.cmd_publisher.publish(movement_command)

    def get_robot_pose(self):
        """Get the robot's current position and orientation"""
        try:
            # Get transform from map to robot
            transform = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            
            # Extract position
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            position = (x, y)
            
            # Extract orientation angle
            quat = transform.transform.rotation
            roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            
            return position, yaw
            
        except TransformException as e:
            self.get_logger().error(f'❌ Cannot find robot position: {e}')
            self.stop_moving()
            return None, None

    def calculate_distance(self, robot_position):
        """Calculate distance from robot to goal"""
        robot_x, robot_y = robot_position
        return np.sqrt((self.goal_x - robot_x)**2 + (self.goal_y - robot_y)**2)

    def normal_navigation(self, robot_position, robot_angle, laser_data):
        """Navigate to goal using potential field forces"""
        # Calculate forces that move the robot
        goal_pull = self.calculate_goal_pull(robot_position)
        obstacle_push = self.calculate_obstacle_push(laser_data)
        
        # Convert forces to movement command
        return self.forces_to_movement_command(goal_pull, obstacle_push, robot_angle)

    def final_orientation_adjustment(self, robot_position, robot_angle, laser_data):
        """
        Final adjustment phase: combine orientation correction with obstacle avoidance
        This prevents the robot from getting stuck while trying to achieve final orientation
        """
        # Calculate normal navigation forces (for obstacle avoidance)
        goal_pull = self.calculate_goal_pull(robot_position)
        obstacle_push = self.calculate_obstacle_push(laser_data)
        
        # Calculate orientation correction force
        orientation_correction = self.calculate_orientation_correction(robot_angle)
        
        # Combine all forces: navigation + orientation correction
        combined_command = self.combine_navigation_and_orientation(
            goal_pull, obstacle_push, orientation_correction, robot_angle
        )
        
        return combined_command

    def calculate_goal_pull(self, robot_position):
        """Calculate how strongly the goal pulls the robot"""
        robot_x, robot_y = robot_position
        distance_to_goal = self.calculate_distance(robot_position)
        
        if distance_to_goal < 0.001:  # Avoid division by zero
            return (0.0, 0.0)
        
        # Direction from robot to goal
        direction_x = (self.goal_x - robot_x) / distance_to_goal
        direction_y = (self.goal_y - robot_y) / distance_to_goal
        
        # Strength of pull toward goal (weaker in final adjustment)
        if self.final_adjustment_mode:
            strength = self.attraction_strength * 0.3  # Reduced in final phase
        else:
            strength = self.attraction_strength
            
        pull_x = strength * direction_x
        pull_y = strength * direction_y
        
        return (pull_x, pull_y)

    def calculate_orientation_correction(self, robot_angle):
        """Calculate force to correct orientation toward final goal angle"""
        angle_error = self.goal_th - robot_angle
        # Keep error between -180 and +180 degrees
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # Check if we're stuck (not making progress)
        if self.last_angle_error is not None:
            if abs(angle_error - self.last_angle_error) < 0.001:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
        self.last_angle_error = angle_error
        
        # If stuck, apply more aggressive correction
        if self.stuck_counter > 10:  # Stuck for 1 second
            self.get_logger().info('⚡ Applying aggressive orientation correction')
            correction_strength = min(1.0, abs(angle_error) * 4.0)
            turn_direction = 1.0 if angle_error > 0 else -1.0
            correction_x = 0.0  # No forward movement from orientation correction
            correction_y = turn_direction * max(0.3, correction_strength)
        else:
            # Normal proportional correction
            correction_x = 0.0
            correction_y = angle_error * self.turn_responsiveness * 0.5
        
        return (correction_x, correction_y)

    def calculate_obstacle_push(self, laser_data):
        """Calculate how obstacles push the robot away"""
        push_x, push_y = 0.0, 0.0
        
        for i, distance in enumerate(laser_data.ranges):
            # Only consider obstacles within influence distance
            if 0.01 < distance < self.obstacle_influence_distance:
                # Angle to this obstacle
                angle = laser_data.angle_min + i * laser_data.angle_increment
                
                # Calculate push strength based on distance
                push_strength = self.calculate_push_strength(distance)
                
                # Push direction is opposite to obstacle direction (away from obstacle)
                push_angle = angle + np.pi  # Add 180 degrees
                
                # Add this obstacle's push to total
                push_x += push_strength * np.cos(push_angle)
                push_y += push_strength * np.sin(push_angle)
                
        return (push_x, push_y)

    def calculate_push_strength(self, distance_to_obstacle):
        """Calculate how strongly to push away from an obstacle"""
        # Closer obstacles push harder
        closeness = (1.0 / distance_to_obstacle - 1.0 / self.obstacle_influence_distance)
        push_force = self.repulsion_strength * closeness * (1.0 / (distance_to_obstacle**2))
        return push_force

    def combine_navigation_and_orientation(self, goal_pull, obstacle_push, orientation_correction, robot_angle):
        """Combine navigation forces with orientation correction"""
        pull_x, pull_y = goal_pull
        push_x, push_y = obstacle_push
        correction_x, correction_y = orientation_correction
        
        # Convert goal pull from map coordinates to robot coordinates
        pull_in_robot_x = (pull_x * np.cos(robot_angle) + pull_y * np.sin(robot_angle))
        pull_in_robot_y = (-pull_x * np.sin(robot_angle) + pull_y * np.cos(robot_angle))
        
        # Combine all forces (all are in robot coordinates now)
        total_x = pull_in_robot_x + push_x + correction_x
        total_y = pull_in_robot_y + push_y + correction_y
        
        # Calculate movement command from combined forces
        return self.calculate_movement_command(total_x, total_y)

    def forces_to_movement_command(self, goal_pull, obstacle_push, robot_angle):
        """Convert attraction and repulsion forces into movement commands"""
        pull_x, pull_y = goal_pull
        push_x, push_y = obstacle_push
        
        # Convert goal pull from map coordinates to robot coordinates
        pull_in_robot_x = (pull_x * np.cos(robot_angle) + pull_y * np.sin(robot_angle))
        pull_in_robot_y = (-pull_x * np.sin(robot_angle) + pull_y * np.cos(robot_angle))
        
        # Combine forces (both are in robot coordinates now)
        total_x = pull_in_robot_x + push_x
        total_y = pull_in_robot_y + push_y
        
        # Calculate movement command
        return self.calculate_movement_command(total_x, total_y)

    def calculate_movement_command(self, total_x, total_y):
        """Calculate movement command from total force vector"""
        # Calculate movement speed
        forward_speed = np.sqrt(total_x**2 + total_y**2)
        forward_speed = min(forward_speed, self.max_forward_speed)
        
        # Calculate turn direction and strength
        desired_turn_angle = np.arctan2(total_y, total_x)
        turn_speed = desired_turn_angle * self.turn_responsiveness
        turn_speed = max(min(turn_speed, self.max_turn_speed), -self.max_turn_speed)
        
        # In final adjustment, prioritize orientation over position
        if self.final_adjustment_mode:
            # Check if orientation is correct
            angle_error = abs(self.last_angle_error) if self.last_angle_error else 0
            if angle_error > self.angle_close_enough:
                # Still need to adjust orientation - reduce forward speed
                forward_speed *= 0.2  # Very slow forward movement
            else:
                # Orientation is good - we're done!
                self.get_logger().info('✅ Goal completely reached! Stopping.')
                self.final_adjustment_mode = False
                return self.create_stop_command()
        
        # Slow down if we need to turn sharply (for normal navigation)
        if not self.final_adjustment_mode:
            if abs(desired_turn_angle) > np.pi / 2:  # More than 90 degrees
                forward_speed *= 0.3
            elif abs(desired_turn_angle) > np.pi / 4:  # More than 45 degrees  
                forward_speed *= 0.7
        
        # Create movement command
        command = Twist()
        command.linear.x = forward_speed
        command.angular.z = turn_speed
        
        return command

    def create_stop_command(self):
        """Create a command to stop the robot"""
        command = Twist()
        command.linear.x = 0.0
        command.angular.z = 0.0
        return command

    def stop_moving(self):
        """Stop the robot completely"""
        try:
            self.cmd_publisher.publish(self.create_stop_command())
        except:
            pass  # Ignore errors during shutdown

def main():
    """Main function to start the planner"""
    rclpy.init()
    planner = PotentialFieldPlanner()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass  # User pressed Ctrl+C
    finally:
        planner.get_logger().info('🛑 Shutting down planner...')
        planner.stop_moving()
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()