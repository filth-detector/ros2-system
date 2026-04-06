#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math

from nav_msgs.msg import Path
from std_msgs.msg import Float64, Empty
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point

class PanTiltController(Node):
    def __init__(self):
        super().__init__('head_controller')
        
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_fov_rad', math.pi / 2.0) 

        self.declare_parameter('cam_offset_y', 0.0) 
        self.declare_parameter('cam_offset_z', 1.1)

        self.declare_parameter('target_distance_m', -1.0) 

        self.declare_parameter('angle_tolerance_rad', 0.02) 
        self.declare_parameter('waypoint_timeout_sec', 2.0) 
        
        self.declare_parameter('yaw_joint_name', 'cleaning_head_yaw_joint')
        self.declare_parameter('pitch_joint_name', 'cleaning_head_pitch_joint')

        self.is_cleaning = False
        self.waypoints_angles = [] 
        self.raw_pixel_waypoints = [] 
        self.current_target_idx = 0
        
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        
        self.time_seeking_waypoint = 0.0
        
        self.hold_position = False

        w = self.get_parameter('image_width').value
        fov = self.get_parameter('camera_fov_rad').value
        self.focal_length = w / (2.0 * math.tan(fov / 2.0))

        self.path_sub = self.create_subscription(Path, '/cleaning_path', self.path_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.pub_pitch = self.create_publisher(Float64, '/cleaning_head_pitch_target_cmd', 10)
        self.pub_yaw = self.create_publisher(Float64, '/cleaning_head_yaw_target_cmd', 10)
        self.pub_aim_pixel = self.create_publisher(Point, '/current_aim_pixel', 10)
        self.pub_done = self.create_publisher(Empty, '/cleaning_done', 10)

        self.srv_start = self.create_service(Trigger, 'start_cleaning', self.start_cleaning_callback)
        self.srv_cancel = self.create_service(Trigger, 'cancel_cleaning', self.cancel_cleaning_callback)
        self.srv_aim = self.create_service(Trigger, 'aim_and_hold', self.aim_and_hold_callback)

        self.dt = 0.02 # 50 Hz
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info("Manual-Depth Pan-Tilt Controller initialized (Direct Pixel Mode).")

    def joint_state_callback(self, msg):
        yaw_name = self.get_parameter('yaw_joint_name').value
        pitch_name = self.get_parameter('pitch_joint_name').value
        
        try:
            if yaw_name in msg.name:
                idx = msg.name.index(yaw_name)
                self.current_yaw = msg.position[idx]
                
            if pitch_name in msg.name:
                idx = msg.name.index(pitch_name)
                self.current_pitch = msg.position[idx]
        except ValueError:
            pass 

    def path_callback(self, msg):
        offset_y = self.get_parameter('cam_offset_y').value
        offset_z = self.get_parameter('cam_offset_z').value
        
        D = self.get_parameter('target_distance_m').value
        
        c_x = self.get_parameter('image_width').value / 2.0
        c_y = self.get_parameter('image_height').value / 2.0
        f = self.focal_length

        self.waypoints_angles = []
        self.raw_pixel_waypoints = [] 
        
        for pose_stamped in msg.poses:
            u_px = pose_stamped.pose.position.x
            v_px = pose_stamped.pose.position.y
            
            self.raw_pixel_waypoints.append((u_px, v_px))
            
            ray_y = (c_x - u_px) / f
            ray_z = (c_y - v_px) / f
            
            target_yaw = math.atan(ray_y)
            target_pitch = -math.atan(ray_z)
            
            self.waypoints_angles.append((target_yaw, target_pitch))
                
        self.get_logger().info(f"Calculated {len(self.waypoints_angles)} joint waypoints.")

    def start_cleaning_callback(self, request, response):
        if len(self.waypoints_angles) == 0:
            response.success = False
            response.message = "No path loaded."
            return response
            
        self.is_cleaning = True
        self.current_target_idx = 0
        self.time_seeking_waypoint = 0.0 
        self.hold_position = False 
        self.get_logger().info("Starting spray sequence (Will return to zero)!")
        
        response.success = True
        response.message = "Cleaning started."
        return response

    def aim_and_hold_callback(self, request, response):
        if len(self.waypoints_angles) == 0:
            response.success = False
            response.message = "No point loaded. Publish a path first."
            return response
            
        self.is_cleaning = True
        self.current_target_idx = 0
        self.time_seeking_waypoint = 0.0 
        self.hold_position = True 
        self.get_logger().info("Aiming sequence started. Position will be held indefinitely.")
        
        response.success = True
        response.message = "Aiming started."
        return response

    def cancel_cleaning_callback(self, request, response):
        self.is_cleaning = False
        self.waypoints_angles = []
        self.raw_pixel_waypoints = []
        self.time_seeking_waypoint = 0.0
        
        self.pub_yaw.publish(Float64(data=0.0))
        self.pub_pitch.publish(Float64(data=0.0))
        
        self.get_logger().info("Sequence canceled. Returning to home position.")
        
        response.success = True
        response.message = "Canceled and returning to home."
        return response

    def control_loop(self):
        if not self.is_cleaning:
            return

        if self.current_target_idx >= len(self.waypoints_angles):
            self.get_logger().info("Sequence completed!")
            
            if not self.hold_position:
                self.pub_yaw.publish(Float64(data=0.0))
                self.pub_pitch.publish(Float64(data=0.0))
                self.get_logger().info("Returning to zero position.")
            else:
                self.get_logger().info("Holding final target position.")
            
            self.is_cleaning = False
            self.time_seeking_waypoint = 0.0
            self.pub_done.publish(Empty())
            return

        if self.current_target_idx < len(self.raw_pixel_waypoints):
            u, v = self.raw_pixel_waypoints[self.current_target_idx]
            aim_msg = Point(x=float(u), y=float(v), z=0.0)
            self.pub_aim_pixel.publish(aim_msg)

        target_yaw, target_pitch = self.waypoints_angles[self.current_target_idx]

        self.pub_yaw.publish(Float64(data=target_yaw))
        self.pub_pitch.publish(Float64(data=target_pitch))

        yaw_error = abs(target_yaw - self.current_yaw)
        pitch_error = abs(target_pitch - self.current_pitch)

        self.get_logger().info(
            f"Idx: {self.current_target_idx:03d} | "
            f"YAW [Tgt: {target_yaw:+.3f}, Cur: {self.current_yaw:+.3f}, Err: {yaw_error:.3f}] | "
            f"PITCH [Tgt: {target_pitch:+.3f}, Cur: {self.current_pitch:+.3f}, Err: {pitch_error:.3f}]"
        )

        tolerance = self.get_parameter('angle_tolerance_rad').value
        timeout = self.get_parameter('waypoint_timeout_sec').value
        
        self.time_seeking_waypoint += self.dt

        if (yaw_error <= tolerance and pitch_error <= tolerance) or (self.time_seeking_waypoint >= timeout):
            if self.time_seeking_waypoint >= timeout:
                self.get_logger().warn(f"Waypoint {self.current_target_idx} timed out! Forcing next.")
                
            self.current_target_idx += 1
            self.time_seeking_waypoint = 0.0 

def main(args=None):
    rclpy.init(args=args)
    controller = PanTiltController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
