#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from cv_bridge import CvBridge
import numpy as np
import cv2

from std_srvs.srv import Trigger

class LawnmowerPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')

        self.declare_parameter('step_size_px', 20) 
        self.declare_parameter('resolution_m_per_px', 0.01) 
        self.declare_parameter('inflation_radius_px', 10) 

        self.br = CvBridge()
        self.latest_mask_msg = None
        self.island_data = None

        self.subscription = self.create_subscription(
            Image, '/segmentation_mask', self.mask_callback, 10
        )
        
        self.path_pub = self.create_publisher(Path, '/cleaning_path', 10)
        self.inflated_mask_pub = self.create_publisher(Image, '/inflated_mask', 10)
        self.centroids_pub = self.create_publisher(PoseArray, '/filth_centroids', 10)
        
        self.srv_blobs = self.create_service(Trigger, 'extract_blobs', self.extract_blobs_callback)
        self.srv_path = self.create_service(Trigger, 'generate_trajectory', self.generate_trajectory_callback)
        
        self.get_logger().info('Cleaning path planner services (Blobs & Trajectory) are ready.')

    def mask_callback(self, msg):
        self.latest_mask_msg = msg

    def extract_blobs_callback(self, request, response):
        self.get_logger().info('Extracting filth blobs...')

        if self.latest_mask_msg is None:
            response.success = False
            response.message = "No segmentation mask has been received yet."
            self.get_logger().warn(response.message)
            return response

        try:
            cv_image = self.br.imgmsg_to_cv2(self.latest_mask_msg, desired_encoding='passthrough')
            
            if len(cv_image.shape) > 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(cv_image, 127, 255, cv2.THRESH_BINARY)
            
            inflation_radius = self.get_parameter('inflation_radius_px').value
            
            if inflation_radius > 0:
                kernel_size = (2 * inflation_radius + 1, 2 * inflation_radius + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
                mask = cv2.dilate(mask, kernel, iterations=1)
                
            inflated_msg = self.br.cv2_to_imgmsg(mask, encoding='mono8')
            inflated_msg.header = self.latest_mask_msg.header
            self.inflated_mask_pub.publish(inflated_msg)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            valid_labels = [i for i in range(1, num_labels)]
            
            if not valid_labels:
                self.island_data = None
                response.success = False
                response.message = "Mask is empty. No blobs extracted."
                self.get_logger().info(response.message)
                return response

            current_pos = np.array([0.0, 0.0]) 
            ordered_labels = []
            unvisited = valid_labels.copy()
            ordered_centroids = []

            while unvisited:
                distances = [np.linalg.norm(centroids[l] - current_pos) for l in unvisited]
                closest_idx = np.argmin(distances)
                closest_label = unvisited.pop(closest_idx)
                
                ordered_labels.append(closest_label)
                current_pos = centroids[closest_label] 
                ordered_centroids.append((current_pos[0], current_pos[1]))

            self.island_data = {
                'mask': mask,
                'labels': labels,
                'stats': stats,
                'ordered_labels': ordered_labels,
                'header': self.latest_mask_msg.header
            }

            resolution = self.get_parameter('resolution_m_per_px').value
            self.publish_centroids(ordered_centroids, resolution, self.latest_mask_msg.header)
            
            response.success = True
            response.message = f"Extracted and ordered {len(ordered_labels)} filth areas."
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to extract blobs: {e}"
            self.get_logger().error(response.message)

        return response

    def generate_trajectory_callback(self, request, response):
        self.get_logger().info('Path generation triggered...')

        if self.island_data is None:
            response.success = False
            response.message = "No blobs extracted yet. Call extract_blobs first."
            self.get_logger().warn(response.message)
            return response

        try:
            step_size = self.get_parameter('step_size_px').value
            resolution = self.get_parameter('resolution_m_per_px').value
            waypoints = []

            labels = self.island_data['labels']
            stats = self.island_data['stats']
            
            for label in self.island_data['ordered_labels']:
                x_start = stats[label, cv2.CC_STAT_LEFT]
                y_start = stats[label, cv2.CC_STAT_TOP]
                width = stats[label, cv2.CC_STAT_WIDTH]
                height = stats[label, cv2.CC_STAT_HEIGHT]

                blob_mask = (labels == label)

                min_y = y_start
                max_y = y_start + height - 1

                going_right = True
                for y in range(min_y, max_y + 1, step_size):
                    row = blob_mask[y, x_start:x_start+width]
                    x_indices_in_bbox = np.where(row)[0]

                    if len(x_indices_in_bbox) == 0:
                        continue

                    min_x = x_start + np.min(x_indices_in_bbox)
                    max_x = x_start + np.max(x_indices_in_bbox)

                    if going_right:
                        waypoints.append((min_x, y))
                        waypoints.append((max_x, y))
                    else:
                        waypoints.append((max_x, y))
                        waypoints.append((min_x, y))

                    going_right = not going_right

            path_msg = self.create_path_msg(waypoints, resolution, self.island_data['header'])
            self.path_pub.publish(path_msg)
            
            response.success = True
            response.message = f"Path generated with {len(waypoints)} waypoints."
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to plan path: {e}"
            self.get_logger().error(response.message)

        return response

    def publish_centroids(self, centroids_px, resolution, header):
        pose_array = PoseArray()
        pose_array.header.frame_id = 'base_link' 
        pose_array.header.stamp = header.stamp
        
        for (x, y) in centroids_px:
            pose = Pose()
            pose.position.x = float(x * resolution)
            pose.position.y = float(y * resolution)
            pose.position.z = 0.0
            
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
            
        self.centroids_pub.publish(pose_array)

    def create_path_msg(self, waypoints, resolution, header):
        path = Path()
        path.header.frame_id = 'base_link' 
        path.header.stamp = header.stamp
        
        for (x, y) in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            
            pose.pose.position.x = float(x * resolution)
            pose.pose.position.y = float(y * resolution)
            pose.pose.position.z = 0.0
            
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
            
        return path

def main(args=None):
    rclpy.init(args=args)
    planner = LawnmowerPlanner()
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
