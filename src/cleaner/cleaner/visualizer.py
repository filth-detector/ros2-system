import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisualizerNode(Node):
    def __init__(self):
        super().__init__('pipeline_visualizer')

        self.declare_parameter('resolution_m_per_px', 0.01)

        self.br = CvBridge()
        
        self.latest_image = None
        self.latest_mask = None
        self.latest_path = None

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/segmentation_mask', self.mask_callback, 10)
        self.create_subscription(Path, '/cleaning_path', self.path_callback, 10)

        self.render_timer = self.create_timer(1.0 / 10.0, self.render_display)
        
        self.get_logger().info('Visualizer node started.')

    def image_callback(self, msg):
        self.latest_image = msg

    def mask_callback(self, msg):
        self.latest_mask = msg

    def path_callback(self, msg):
        self.latest_path = msg

    def render_display(self):
        if self.latest_image is None:
            return

        try:
            # Raw image
            display_img = self.br.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')

            # Segmentation mask
            if self.latest_mask is not None:
                mask = self.br.imgmsg_to_cv2(self.latest_mask, desired_encoding='mono8')
                
                overlay = np.zeros_like(display_img)
                overlay[mask > 0] = [0, 0, 255] # BGR
                
                display_img = cv2.addWeighted(display_img, 1.0, overlay, 0.3, 0)

            # Cleaning path
            if self.latest_path is not None:
                resolution = self.get_parameter('resolution_m_per_px').value
                waypoints = []
                
                for pose_stamped in self.latest_path.poses:
                    px = int(pose_stamped.pose.position.x / resolution)
                    py = int(pose_stamped.pose.position.y / resolution)
                    waypoints.append([px, py])

                if len(waypoints) > 1:
                    pts = np.array(waypoints, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            cv2.imshow("Cleaning pipeline visualizer", display_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error rendering: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
