import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np

class LawnmowerPlanner(Node):
    def __init__(self):
        super().__init__('cleaning_path_planner')

        self.declare_parameter('step_size_px', 20) 
        self.declare_parameter('resolution_m_per_px', 0.01) 

        self.subscription = self.create_subscription(
            Image,
            '/segmentation_mask',
            self.mask_callback,
            10
        )
        self.path_pub = self.create_publisher(Path, '/cleaning_path', 10)
        
        self.br = CvBridge()
        self.get_logger().info('Cleaning path planner started.')

    def mask_callback(self, msg):
        try:
            mask = self.br.imgmsg_to_cv2(msg, desired_encoding='mono8')
            
            step_size = self.get_parameter('step_size_px').value
            resolution = self.get_parameter('resolution_m_per_px').value
            
            waypoints_px = self.generate_path(mask, step_size)
            
            path_msg = self.create_path_msg(waypoints_px, resolution, msg.header)
            
            self.path_pub.publish(path_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to plan path: {e}')

    def generate_path(self, mask, step_size):
        waypoints = []
        
        # Find all Y and X coordinates where the mask is active
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return waypoints
            
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        
        going_right = True
        for y in range(min_y, max_y + 1, step_size):
            row_x = x_indices[y_indices == y]
            
            if len(row_x) == 0:
                continue
                
            min_x, max_x = np.min(row_x), np.max(row_x)
            
            if going_right:
                waypoints.append((min_x, y))
                waypoints.append((max_x, y))
            else:
                waypoints.append((max_x, y))
                waypoints.append((min_x, y))
                
            going_right = not going_right
            
        return waypoints

    def create_path_msg(self, waypoints, resolution, header):
        path = Path()
        path.header = header
      
        for (x, y) in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            
            pose.pose.position.x = float(x * resolution)
            pose.pose.position.y = float(y * resolution)
            pose.pose.position.z = 0.0
            
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
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
