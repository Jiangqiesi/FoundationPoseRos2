import sys

sys.path.append("./FoundationPose")
sys.path.append("./FoundationPose/nvdiffrast")

import rclpy
from rclpy.node import Node
from estimater import *
import cv2
import numpy as np
import trimesh
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from cv_bridge import CvBridge
import argparse
import os
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM
from cam_2_base_transform import *
import os
import tkinter as tk
from tkinter import Listbox, END, Button
import glob
import yaml

# Save the original `__init__` and `register` methods
original_init = FoundationPose.__init__
original_register = FoundationPose.register


# Modify `__init__` to add `is_register` attribute
def modified_init(
    self,
    model_pts,
    model_normals,
    symmetry_tfs=None,
    mesh=None,
    scorer=None,
    refiner=None,
    glctx=None,
    debug=0,
    debug_dir="./FoundationPose",
):
    original_init(
        self,
        model_pts,
        model_normals,
        symmetry_tfs,
        mesh,
        scorer,
        refiner,
        glctx,
        debug,
        debug_dir,
    )
    self.is_register = False  # Initialize as False


# Modify `register` to set `is_register` to True when a pose is registered
def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True  # Set to True after registration
    return pose


# Apply the modifications
FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOperatorIssue=false, reportArgumentType=false, reportMissingTypeArgument=false

from typing import Dict, List, Optional, Sequence, Tuple

from foundationpose_msgs.srv import AssignModels, PerceptionStatus, Resegment


def str2bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def collect_mesh_files(mesh_dir: str) -> List[str]:
    resolved_dir = mesh_dir
    if not os.path.isabs(resolved_dir):
        resolved_dir = os.path.join(code_dir, resolved_dir)

    mesh_paths = (
        glob.glob(os.path.join(resolved_dir, "**", "*.obj"), recursive=True)
        + glob.glob(os.path.join(resolved_dir, "**", "*.stl"), recursive=True)
        + glob.glob(os.path.join(resolved_dir, "**", "*.STL"), recursive=True)
    )
    mesh_paths.sort()
    return mesh_paths


def masks_from_sam_result(result: object, height: int, width: int) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    if result is None:
        return masks

    mask_data = getattr(result, "masks", None)
    if mask_data is None or getattr(mask_data, "xy", None) is None:
        return masks

    for contour_xy in mask_data.xy:
        contour = np.asarray(contour_xy, dtype=np.int32).reshape(-1, 1, 2)
        if contour.shape[0] < 3:
            continue
        mask = np.zeros((height, width), np.uint8)
        _ = cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        if int(mask.sum()) == 0:
            continue
        masks.append(mask)

    return masks


def pose_to_pose_array(center_pose: np.ndarray) -> np.ndarray:
    position = center_pose[:3, 3]
    rotation_matrix = center_pose[:3, :3]
    quaternion_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return np.concatenate((position, quaternion_xyzw))


parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument("--est_refine_iter", type=int, default=4)
parser.add_argument("--track_refine_iter", type=int, default=2)
parser.add_argument(
    "--base_frame", type=str, default="base_link", help="发布位姿使用的基坐标系名称"
)
parser.add_argument(
    "--scale", type=float, default=0.001, help="模型缩放系数，默认0.001（将mm转换为m）"
)
parser.add_argument(
    "--auto-assign",
    type=str2bool,
    default=True,
    help="是否在分割后按面积排序自动分配模型，默认True",
)
parser.add_argument(
    "--mesh-dir", type=str, default="demo_data", help="网格目录，默认 demo_data"
)
args = parser.parse_args()


class PoseEstimationNode(Node):
    def __init__(self, mesh_paths: Sequence[str]):
        super().__init__("foundationpose_perception_server")

        self._auto_assign = bool(args.auto_assign)
        self.base_frame_id = args.base_frame

        self.image_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            10,
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,
            10,
        )

        self.bridge = CvBridge()
        self.depth_image: Optional[np.ndarray] = None
        self.color_image: Optional[np.ndarray] = None
        self.cam_K: Optional[np.ndarray] = None

        self.mesh_files = list(mesh_paths)
        model_scale = args.scale
        self.meshes = [trimesh.load(mesh) for mesh in self.mesh_files]
        for mesh in self.meshes:
            mesh.apply_scale(model_scale)

        self.bboxes = [mesh.bounds for mesh in self.meshes]

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        self.seg_model = SAM("sam2.1_b.pt")

        self.pose_estimations: Dict[int, Dict[str, object]] = {}
        self.pose_publishers: Dict[str, object] = {}
        self.grasp_array_publishers: Dict[str, object] = {}
        self.best_grasp_publishers: Dict[str, object] = {}
        self.tracked_objects: List[np.ndarray] = []
        self._segmentation_done = False
        self._last_masks: List[np.ndarray] = []
        self._last_color_shape: Optional[Tuple[int, int]] = None
        self._latest_assign_summary = "未分割"
        self._frame_counter = 0

        self.resegment_srv = self.create_service(
            Resegment,
            "/perception/resegment",
            self.handle_resegment,
        )
        self.assign_models_srv = self.create_service(
            AssignModels,
            "/perception/assign_models",
            self.handle_assign_models,
        )
        self.status_srv = self.create_service(
            PerceptionStatus,
            "/perception/get_status",
            self.handle_get_status,
        )

        self.get_logger().info(
            f"感知服务端已启动，mesh数量: {len(self.mesh_files)} auto_assign={self._auto_assign}"
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if self.cam_K is None:
            self.cam_K = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(f"Camera intrinsic matrix initialized: {self.cam_K}")

    def image_callback(self, msg: Image) -> None:
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def depth_callback(self, msg: Image) -> None:
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1") / 1e3
        self.process_images()

    def data_ready(self) -> bool:
        return (
            self.color_image is not None
            and self.depth_image is not None
            and self.cam_K is not None
        )

    def _prepare_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
        if not self.data_ready():
            return None

        assert self.color_image is not None
        assert self.depth_image is not None
        h, w = self.color_image.shape[:2]
        color = cv2.resize(self.color_image, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(self.depth_image, (w, h), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.1) | (depth >= np.inf)] = 0
        return color, depth, h, w

    def _segment_once(self, color: np.ndarray, h: int, w: int) -> List[np.ndarray]:
        prediction = self.seg_model.predict(color, verbose=False)
        if not prediction:
            return []
        masks = masks_from_sam_result(prediction[0], h, w)
        if not masks:
            return []
        masks.sort(key=lambda m: int(np.count_nonzero(m)), reverse=True)
        return masks

    def _reset_tracking_state(self) -> None:
        self.pose_estimations.clear()
        self.tracked_objects = []
        self._segmentation_done = False

    def _trigger_resegment(self) -> None:
        self._reset_tracking_state()
        self._last_masks = []
        self._latest_assign_summary = "等待重新分割"

    def _assign_models(
        self,
        mesh_paths: Sequence[str],
        mask_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[bool, str]:
        if not self._last_masks:
            return False, "当前没有可用掩码，请先完成分割"

        if not mesh_paths:
            return False, "mesh_paths 为空"

        index_list: List[int] = []
        if mask_indices is None:
            upper = min(len(self._last_masks), len(mesh_paths))
            index_list = list(range(upper))
        else:
            index_list = [int(i) for i in mask_indices]

        if len(index_list) != len(mesh_paths):
            return (
                False,
                f"mesh_paths({len(mesh_paths)}) 与 mask_indices({len(index_list)}) 数量不一致",
            )

        for i in index_list:
            if i < 0 or i >= len(self._last_masks):
                return (
                    False,
                    f"mask index 越界: {i}, 可用范围 [0, {len(self._last_masks) - 1}]",
                )

        self.pose_estimations.clear()
        self.tracked_objects = []
        normalized_mesh_files = [os.path.abspath(p) for p in self.mesh_files]

        for obj_idx, (mesh_path, mask_idx) in enumerate(
            zip(mesh_paths, index_list), start=1
        ):
            mesh_path_abs = os.path.abspath(mesh_path)
            if mesh_path_abs not in normalized_mesh_files:
                return False, f"未在启动网格列表中找到: {mesh_path}"

            mesh_file_index = normalized_mesh_files.index(mesh_path_abs)
            mesh = self.meshes[mesh_file_index]
            mask = self._last_masks[mask_idx]

            pose_est = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=self.scorer,
                refiner=self.refiner,
                glctx=self.glctx,
            )

            self.pose_estimations[obj_idx] = {
                "pose_est": pose_est,
                "mask": mask,
                "mesh_path": self.mesh_files[mesh_file_index],
                "mask_index": mask_idx,
                "bbox": self.bboxes[mesh_file_index],
            }
            self.tracked_objects.append(mask)

        self._segmentation_done = True
        ids = sorted(self.pose_estimations.keys())
        self._latest_assign_summary = f"分配完成，跟踪对象ID: {ids}"
        self.get_logger().info(self._latest_assign_summary)
        return True, self._latest_assign_summary

    def _auto_assign_models(self) -> Tuple[bool, str]:
        if not self._last_masks:
            return False, "无可分配掩码"
        if not self.mesh_files:
            return False, "无可分配网格文件"

        assign_count = min(len(self.mesh_files), len(self._last_masks))
        selected_mesh = self.mesh_files[:assign_count]
        selected_masks = list(range(assign_count))
        ok, message = self._assign_models(selected_mesh, selected_masks)

        if ok and len(self._last_masks) != len(self.mesh_files):
            self.get_logger().warn(
                f"掩码数量({len(self._last_masks)}) 与网格数量({len(self.mesh_files)}) 不一致，"
                f"仅分配前 {assign_count} 个"
            )
        return ok, message

    def _ensure_segmented_and_assigned(self, color: np.ndarray, h: int, w: int) -> bool:
        if self._segmentation_done:
            return True
        if (not self._auto_assign) and self._last_masks:
            return False

        masks = self._segment_once(color, h, w)
        if not masks:
            self.get_logger().warn("SAM2 未检测到掩码，等待下一帧重试")
            self._last_masks = []
            return False

        self._last_masks = masks
        self._last_color_shape = (h, w)
        self.get_logger().info(f"SAM2 分割完成，掩码数量: {len(self._last_masks)}")

        if not self._auto_assign:
            self._latest_assign_summary = (
                "分割完成，等待 /perception/assign_models 手动分配"
            )
            self.get_logger().info(self._latest_assign_summary)
            return False

        ok, message = self._auto_assign_models()
        if not ok:
            self.get_logger().error(f"自动分配失败: {message}")
            return False

        return True

    def process_images(self) -> None:
        prepared = self._prepare_frame()
        if prepared is None:
            return

        color, depth, h, w = prepared
        if not self._ensure_segmented_and_assigned(color, h, w):
            return

        for idx, data in self.pose_estimations.items():
            pose_est = data["pose_est"]
            obj_mask = data["mask"]

            if pose_est.is_register:
                pose = pose_est.track_one(
                    rgb=color,
                    depth=depth,
                    K=self.cam_K,
                    iteration=args.track_refine_iter,
                )
                center_pose = pose
                self.publish_pose_stamped(
                    center_pose=center_pose,
                    frame_id=f"object_{idx}_frame",
                    topic_name=f"/Current_OBJ_position_{idx}",
                    obj_idx=idx,
                )
            else:
                pose_est.register(
                    K=self.cam_K,
                    rgb=color,
                    depth=depth,
                    ob_mask=obj_mask,
                    iteration=args.est_refine_iter,
                )

        self._frame_counter += 1

    def visualize_pose(
        self, image: np.ndarray, center_pose: np.ndarray, idx: int
    ) -> np.ndarray:
        data = self.pose_estimations.get(idx)
        if data is None:
            return image
        bbox = data["bbox"]
        vis = draw_posed_3d_box(self.cam_K, img=image, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self.cam_K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        return vis

    def publish_pose_stamped(
        self, center_pose: np.ndarray, frame_id: str, topic_name: str, obj_idx: int
    ) -> None:
        del frame_id, obj_idx
        if topic_name not in self.pose_publishers:
            self.pose_publishers[topic_name] = self.create_publisher(
                PoseStamped, topic_name, 10
            )

        pose_stamped_msg = PoseStamped()
        pose_stamped_msg.header.stamp = self.get_clock().now().to_msg()
        pose_stamped_msg.header.frame_id = self.base_frame_id

        pose_array = pose_to_pose_array(center_pose)
        transformed_pose = transformation(pose_array)

        pose_stamped_msg.pose.position.x = float(transformed_pose[0])
        pose_stamped_msg.pose.position.y = float(transformed_pose[1])
        pose_stamped_msg.pose.position.z = float(transformed_pose[2])

        pose_stamped_msg.pose.orientation.w = float(transformed_pose[3])
        pose_stamped_msg.pose.orientation.x = float(transformed_pose[4])
        pose_stamped_msg.pose.orientation.y = float(transformed_pose[5])
        pose_stamped_msg.pose.orientation.z = float(transformed_pose[6])

        self.pose_publishers[topic_name].publish(pose_stamped_msg)

    def handle_resegment(
        self, request: Resegment.Request, response: Resegment.Response
    ) -> Resegment.Response:
        force = bool(request.force)
        self._trigger_resegment()

        if force and self.data_ready():
            prepared = self._prepare_frame()
            if prepared is not None:
                color, _depth, h, w = prepared
                if self._ensure_segmented_and_assigned(color, h, w):
                    response.success = True
                    response.message = "已立即重新分割并完成分配"
                    return response

        response.success = True
        response.message = "已重置分割状态，下一帧执行重新分割"
        return response

    def handle_assign_models(
        self,
        request: AssignModels.Request,
        response: AssignModels.Response,
    ) -> AssignModels.Response:
        mesh_paths = list(request.mesh_paths)
        mask_indices = [int(v) for v in request.mask_indices]

        if not self._last_masks:
            if self.data_ready():
                prepared = self._prepare_frame()
                if prepared is not None:
                    color, _depth, h, w = prepared
                    self._last_masks = self._segment_once(color, h, w)
            if not self._last_masks:
                response.success = False
                response.message = "当前无分割结果，请先确保相机数据可用"
                return response

        ok, message = self._assign_models(mesh_paths, mask_indices)
        response.success = ok
        response.message = message
        return response

    def handle_get_status(
        self,
        request: PerceptionStatus.Request,
        response: PerceptionStatus.Response,
    ) -> PerceptionStatus.Response:
        del request
        response.ready = self.data_ready()
        response.segmentation_done = self._segmentation_done
        object_ids = sorted(self.pose_estimations.keys())
        response.num_tracked_objects = len(object_ids)
        response.tracked_object_ids = [int(i) for i in object_ids]
        return response


def main(main_args: Optional[Sequence[str]] = None) -> None:
    mesh_paths = collect_mesh_files(args.mesh_dir)
    if not mesh_paths:
        raise RuntimeError(f"未找到网格文件: {args.mesh_dir}")

    rclpy.init(args=main_args)
    node = PoseEstimationNode(mesh_paths)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
