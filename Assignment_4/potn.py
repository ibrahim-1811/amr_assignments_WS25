import pybullet as p
import numpy as np
import time

# Constants
k_a = 0.5  # attractive force constant
k_r = 1.0  # repulsive force constant
rho_0 = 2.0  # threshold distance for repulsive force
goal_threshold_distance = 0.1  # distance threshold to consider reaching the goal
orientation_threshold_distance = 3.0  # distance threshold for orientation control
time_step = 0.1  # simulation time step

# Goal pose
goal_pose = np.array([4.0, 10.0, -1.0])

# Function to calculate attractive force
def attractive_force(current_pose, goal_pose):
    return -k_a * (current_pose - goal_pose) / np.linalg.norm(current_pose - goal_pose)

# Function to calculate repulsive force from a single obstacle
def repulsive_force(current_pose, obstacle_pose):
    distance = np.linalg.norm(current_pose - obstacle_pose)
    
    if distance < rho_0:
        direction = (current_pose - obstacle_pose) / distance
        return k_r * ((1 / distance) - (1 / rho_0)) * (1 / distance**2) * direction
    else:
        return np.zeros_like(current_pose)

# Function to calculate total force
def calculate_total_force(current_pose, goal_pose, obstacles):
    total_force = attractive_force(current_pose, goal_pose)
    
    for obstacle_pose in obstacles:
        total_force += repulsive_force(current_pose, obstacle_pose)
    
    return total_force

# Initialize simulation
p.connect(p.GUI)  # Use p.DIRECT for non-graphical simulation
p.setGravity(0, 0, -9.81)
p.setTimeStep(time_step)

# Create robot and obstacles (simplified as spheres)
robot_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=1.0)
robot_position = [0, 0, 0.5]
robot_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot_body = p.createMultiBody(baseCollisionShapeIndex=robot_id,
                               basePosition=robot_position,
                               baseOrientation=robot_orientation)

obstacle_ids = []
obstacle_positions = [[2, 5, 0.5], [6, 7, 0.5]]
for obstacle_position in obstacle_positions:
    obstacle_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    obstacle_body = p.createMultiBody(baseCollisionShapeIndex=obstacle_id,
                                      basePosition=obstacle_position)
    obstacle_ids.append(obstacle_body)

# Main simulation loop
for _ in range(1000):  # Run for 1000 simulation steps (adjust as needed)
    # Get robot's current pose
    robot_position, robot_orientation = p.getBasePositionAndOrientation(robot_body)
    current_pose = np.array(robot_position + p.getEulerFromQuaternion(robot_orientation))

    # Calculate total force
    total_force = calculate_total_force(current_pose[:2], goal_pose[:2], obstacle_positions)

    # Update robot's position based on the total force
    new_robot_position = robot_position + np.append(total_force, [0]) * time_step
    p.resetBasePositionAndOrientation(robot_body, new_robot_position, robot_orientation)

    # Optional: Adjust orientation towards the goal until the robot is close enough
    if np.linalg.norm(current_pose[:2] - goal_pose[:2]) > orientation_threshold_distance:
        desired_theta = np.arctan2(goal_pose[1] - current_pose[1], goal_pose[0] - current_pose[0])
        p.resetBasePositionAndOrientation(robot_body, new_robot_position,
                                          p.getQuaternionFromEuler([0, 0, desired_theta]))

    # Optional: Add control to limit angular velocity for smoother orientation adjustments

    # Step simulation
    p.stepSimulation()
    time.sleep(time_step)

# Close the simulation
p.disconnect()
