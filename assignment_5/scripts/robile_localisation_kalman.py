#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from robile_interfaces.msg import PositionLabelledArray
from nav_msgs.msg import Odometry
import numpy as np
import tf2_ros
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import time

class LocalisationUsingKalmanFilter(Node):
    """
    Landmark based localisation using Kalman Filter
    This is a partially structured class for AMR assignment
    """

    def __init__(self):
        super().__init__('localisation_using_kalman_filter')

        # declaring and getting parameters from yaml file
        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_frame', 'map'),
                ('odom_frame', 'odom'),                
                ('laser_link_frame', 'base_laser_front_link'),
                ('real_base_link_frame', 'real_base_link'),
                ('scan_topic', 'scan'),
                ('odom_topic', 'odom'),
                ('rfid_tag_poses_topic', 'rfid_tag_poses'),
                ('initial_pose_topic', 'initialpose'),
                ('real_base_link_pose_topic', 'real_base_link_pose'),
                ('estimated_base_link_pose_topic', 'estimated_base_link_pose'),
                ('minimum_travel_distance', 0.1),
                ('minimum_travel_heading', 0.1),
                ('rfid_tags.A', [0.,0.]),
                ('rfid_tags.B', [0.,0.]),
                ('rfid_tags.C', [0.,0.]),
                ('rfid_tags.D', [0.,0.]),
                ('rfid_tags.E', [0.,0.]),                        
            ])

        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.laser_link_frame = self.get_parameter('laser_link_frame').get_parameter_value().string_value
        self.real_base_link_frame = self.get_parameter('real_base_link_frame').get_parameter_value().string_value
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.rfid_tag_poses_topic = self.get_parameter('rfid_tag_poses_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.real_base_link_pose_topic = self.get_parameter('real_base_link_pose_topic').get_parameter_value().string_value
        self.estimated_base_link_pose_topic = self.get_parameter('estimated_base_link_pose_topic').get_parameter_value().string_value
        self.minimum_travel_distance = self.get_parameter('minimum_travel_distance').get_parameter_value().double_value
        self.minimum_travel_heading = self.get_parameter('minimum_travel_heading').get_parameter_value().double_value
        self.rfid_tags_A = self.get_parameter('rfid_tags.A').get_parameter_value().double_array_value
        self.rfid_tags_B = self.get_parameter('rfid_tags.B').get_parameter_value().double_array_value
        self.rfid_tags_C = self.get_parameter('rfid_tags.C').get_parameter_value().double_array_value
        self.rfid_tags_D = self.get_parameter('rfid_tags.D').get_parameter_value().double_array_value
        self.rfid_tags_E = self.get_parameter('rfid_tags.E').get_parameter_value().double_array_value
        self.lastOdometry = None

        # setting up laser scan and rfid tag subscribers
        self.rfid_tag_subscriber = self.create_subscription(PositionLabelledArray, self.rfid_tag_poses_topic, self.rfid_callback, 10)
        self.real_laser_link_subscriber = self.create_subscription(PoseStamped, self.real_base_link_pose_topic, self.real_base_link_pose_callback, 10)        
        self.odom_subscriber = self.create_subscription(Odometry, self.odom_topic, self.updateBelief, 10)
        self.estimated_robot_pose_publisher = self.create_publisher(PoseStamped, self.estimated_base_link_pose_topic, 10)
        
        # setting up tf2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.state = np.array([0.,0.,0.]) # This is x0 which contains x,y,theta
        self.cov = np.eye(3,3)*1e-8
        self.mutexUpdate = False

        # In homogeneous coordinates
        self.odomRFID = {
            "A": np.array([1.0, 1.0, 1.0]),
            "B": np.array([6.0, 1.0, 1.0]),
            "C": np.array([3.0, -1.0, 1.0]),
            "D": np.array([1.0, -3.0, 1.0]),
            "E": np.array([4.0, -4.0, 1.0])
        }

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def getTransformation(self):
        x,y,theta = self.state
        s = np.sin(theta)
        c = np.cos(theta)
        return np.array([
            [c,-s,x],
            [s,c,y],
            [0,0,1],
        ])
    
    def quatToAngle(self, quat):
        return euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]
    
    def extractSubCov(self, cov):
        return cov.reshape(6,6)[(0,1,5), :][:, (0,1,5)]
    
    def getBeliefRFIDError(self, detectedRFID):
        tranformation = self.getTransformation()

        idToIdx = {"A":0,"B":1,"C":2,"D":3,"E":4}
        beliefRFIDError = np.zeros((6,1))
        for el in detectedRFID.positions:
            idx = idToIdx[el.name]
            transformedPoint = np.dot(tranformation,self.odomRFID[el.name].T).T
            #6x1
            beliefRFIDError[idx] = np.sqrt((el.position.x - transformedPoint[0])**2 + (el.position.y - transformedPoint[1])**2)

        return beliefRFIDError

    def computeKalmanGain(self):
        H = np.ones((6,3))
        R = np.eye(6,6) * 0.01
        t1 = np.matmul(self.cov, H.T)
        t2 = np.matmul(H,np.matmul(self.cov, H.T))+R
        K = np.matmul(t1, np.linalg.inv(t2))
        return K 

    def rfid_callback(self, msg: PositionLabelledArray):
        """
        Based on the detected RFID tags, performing measurement update
        """        
        if not self.mutexUpdate:
            return
        
        self.mutexUpdate = False
        poseEstimation = PoseStamped()

        K = self.computeKalmanGain()

        be = self.getBeliefRFIDError(msg)

        self.state += np.dot(K,be)[:,0]

        self.cov = np.dot((np.eye(3,3) - np.dot(K, np.ones((6,3)))), self.cov)

        poseEstimation.header.stamp = self.get_clock().now().to_msg()
        poseEstimation.header.frame_id = self.map_frame
        poseEstimation.pose.position.x = self.state[0]
        poseEstimation.pose.position.y = self.state[1]
        poseEstimation.pose.position.z = 0.
        quat = quaternion_from_euler(0,0,self.state[2])
        poseEstimation.pose.orientation.x = quat[0]
        poseEstimation.pose.orientation.y = quat[1]
        poseEstimation.pose.orientation.z = quat[2]
        poseEstimation.pose.orientation.w = quat[3]

        print("LINK POSE REAL", self.real_laser_link_pose)
        print("STATE", self.state)

        self.estimated_robot_pose_publisher.publish(poseEstimation)

    def real_base_link_pose_callback(self, msg):
        """
        Updating the base_link pose based on the update in robile_rfid_tag_finder.py
        """
        yaw = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])[2]
        self.real_laser_link_pose = [msg.pose.position.x, msg.pose.position.y, yaw]

    def updateBelief(self, msg:Odometry):
        if not self.lastOdometry:
            self.lastOdometry = msg
            return 

        linearOfft = np.array([msg.pose.pose.position.x - self.lastOdometry.pose.pose.position.x, 
                               msg.pose.pose.position.y - self.lastOdometry.pose.pose.position.y ])
        angularOfft = self.quatToAngle(msg.pose.pose.orientation) - self.quatToAngle(self.lastOdometry.pose.pose.orientation)

        if np.linalg.norm(linearOfft) < self.minimum_travel_distance and abs(angularOfft) < self.minimum_travel_heading:
            return
        
        covOfft = self.extractSubCov(msg.pose.covariance) - self.extractSubCov(self.lastOdometry.pose.covariance)

        self.state += np.append(linearOfft, angularOfft)
        self.cov += covOfft
        self.lastOdometry = msg

        self.mutexUpdate = True


def main(args=None):
    rclpy.init(args=args)

    try:
        localisation_using_kalman_filter = LocalisationUsingKalmanFilter()
        rclpy.spin(localisation_using_kalman_filter)

    finally:
        localisation_using_kalman_filter.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()