import yaml
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener

class FoundationPoseConfig:
    def __init__(self, config_path="config/pick_place.yaml"):
        # Resolve path relative to package root if needed, but here we assume CWD or absolute
        if not os.path.isabs(config_path):
            # Attempt to find it relative to the script location or CWD
            # specific to this environment structure
            base_dir = os.path.dirname(os.path.abspath(__file__))
            possible_path = os.path.join(base_dir, config_path)
            if os.path.exists(possible_path):
                config_path = possible_path

        self.config_path = config_path
        self.data = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            print(f"[Config] Warning: Config file not found at {self.config_path}")
            return {}
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Config] Error loading config: {e}")
            return {}

    @property
    def base_frame(self):
        return self.data.get('frames', {}).get('base_frame', 'base_link')

    @property
    def camera_frame(self):
        return self.data.get('frames', {}).get('camera_frame', 'camera_color_optical_frame')

    @property
    def pose_topic_prefix(self):
        return self.data.get('topics', {}).get('pose_topic_prefix', '/Current_OBJ_position_')

    @property
    def pose_max_age_sec(self):
        return self.data.get('motion', {}).get('pose_max_age_sec', 0.5)

    @property
    def motion_params(self):
        return self.data.get('motion', {})

    def get_object_offset(self, source_id):
        # Returns [x,y,z] in target frame
        obj_data = self.data.get('objects', {}).get(str(source_id), {})
        return obj_data.get('place_offset_local', None)

class TFHandler:
    def __init__(self, node, base_frame, camera_frame):
        self.node = node
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        # Hardcoded fallback from cam_2_base_transform.py (converted to Matrix)
        # Translation: [0.48782276, 0.31000138, 0.64002418]
        # Rotation (xyzw): [0.42074526, 0.84710941, -0.31232969, -0.08848301]
        # Note: The original file had w at index 3? No, R.from_quat expects xyzw.
        # Original code: p_CwrtB.orientation.w = -0.08848301 ...
        self.fallback_T_base_cam = np.eye(4)
        r = R.from_quat([0.42074526, 0.84710941, -0.31232969, -0.08848301])
        self.fallback_T_base_cam[:3, :3] = r.as_matrix()
        self.fallback_T_base_cam[:3, 3] = [0.48782276, 0.31000138, 0.64002418]

    def get_transform_matrix(self):
        try:
            # Lookup T_base_cam (transform from camera to base)
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time())

            tr = t.transform.translation
            rot = t.transform.rotation

            T = np.eye(4)
            T[:3, 3] = [tr.x, tr.y, tr.z]
            r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
            T[:3, :3] = r.as_matrix()
            return T
        except Exception:
            # self.node.get_logger().warn(f"TF lookup failed, using fallback")
            return self.fallback_T_base_cam

    def transform_pose(self, pose_matrix_cam):
        # pose_matrix_cam: 4x4 matrix of object in camera frame
        # returns: 4x4 matrix of object in base frame
        T_base_cam = self.get_transform_matrix()
        return T_base_cam @ pose_matrix_cam
