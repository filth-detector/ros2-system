import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10 # QoS profile depth
        )

        self.br = CvBridge()
        self.get_logger().info('Image subscriber node started.')

    def image_callback(self, msg):
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            cv2.imshow("Camera Feed", cv_image)
            cv2.waitKey(1) # Required to update the GUI window

        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
