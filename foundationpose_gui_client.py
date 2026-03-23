#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FoundationPose PySide6 图形客户端。"""

import argparse
import glob
import os
import sys
from enum import IntEnum
from typing import Any, Callable, cast, final

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from foundationpose_client import FoundationPoseClient


class GUIState(IntEnum):
    IDLE = 0
    MASKS_DISPLAYED = 1
    ASSIGNING = 2
    TRACKING = 3


def collect_mesh_files(mesh_dir: str) -> list[str]:
    mesh_paths = (
        glob.glob(os.path.join(mesh_dir, "**", "*.obj"), recursive=True)
        + glob.glob(os.path.join(mesh_dir, "**", "*.stl"), recursive=True)
        + glob.glob(os.path.join(mesh_dir, "**", "*.STL"), recursive=True)
    )
    mesh_paths.sort()
    return mesh_paths


@final
class ServiceWorker(QThread):
    result_ready: Signal = Signal(object)
    error_ready: Signal = Signal(str)

    def __init__(
        self,
        fn: Callable[..., dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self._fn: Callable[..., dict[str, Any]] = fn
        self._args: tuple[Any, ...] = args
        self._kwargs: dict[str, Any] = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_ready.emit(str(e))


@final
class ClickableImageLabel(QLabel):
    clicked: Signal = Signal(object)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(ev)
        super().mousePressEvent(ev)


@final
class FoundationPoseGUI(QMainWindow):
    def __init__(self, mesh_dir: str = "demo_data"):
        super().__init__()
        self.setWindowTitle("FoundationPose GUI Client")
        self.resize(1280, 780)

        self.client = FoundationPoseClient(node_name="foundationpose_gui_client")
        self.state = GUIState.IDLE
        self.session_id: int = 0
        self.num_masks: int = 0
        self.assignments: dict[int, str] = {}
        self.mesh_dir = mesh_dir
        self.mesh_paths = collect_mesh_files(mesh_dir)

        self._latest_image: npt.NDArray[np.uint8] | None = None
        self._workers: list[ServiceWorker] = []

        self._init_ui()
        self._load_mesh_list()
        self._update_assignment_display()
        self._update_status_bar()

        self.update_timer = QTimer(self)
        self.update_timer.setInterval(30)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start()

    def _init_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        button_layout = QHBoxLayout()
        self.resegment_btn = QPushButton("Resegment")
        self.confirm_btn = QPushButton("Confirm Assignment")
        self.status_btn = QPushButton("Status")
        self.reset_btn = QPushButton("Reset")

        button_layout.addWidget(self.resegment_btn)
        button_layout.addWidget(self.confirm_btn)
        button_layout.addWidget(self.status_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch(1)
        root_layout.addLayout(button_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.image_label = ClickableImageLabel("等待图像...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(700, 500)
        self.image_label.setStyleSheet(
            "QLabel { background-color: #111111; color: #dddddd; }"
        )
        splitter.addWidget(self.image_label)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.mesh_list = QListWidget()
        self.mesh_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        right_layout.addWidget(QLabel("可用模型"))
        right_layout.addWidget(self.mesh_list, 1)

        assignment_group = QGroupBox("当前分配")
        assignment_layout = QVBoxLayout(assignment_group)
        self.assignment_label = QLabel("暂无分配")
        self.assignment_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.assignment_label.setWordWrap(True)
        assignment_layout.addWidget(self.assignment_label)
        right_layout.addWidget(assignment_group)

        splitter.addWidget(right_panel)
        splitter.setSizes([900, 380])
        root_layout.addWidget(splitter, 1)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        _ = self.resegment_btn.clicked.connect(self.on_resegment)
        _ = self.confirm_btn.clicked.connect(self.on_confirm)
        _ = self.status_btn.clicked.connect(self.on_status)
        _ = self.reset_btn.clicked.connect(self.on_reset)
        _ = self.image_label.clicked.connect(self.on_image_click)

    def _load_mesh_list(self) -> None:
        self.mesh_list.clear()
        for path in self.mesh_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.mesh_list.addItem(item)
        self.status_bar.showMessage(f"已加载 {len(self.mesh_paths)} 个模型", 3000)

    def _run_worker(
        self,
        fn: Callable[..., dict[str, Any]],
        on_success: Callable[[dict[str, Any]], None],
        action_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        worker = ServiceWorker(fn, *args, **kwargs)
        self._workers.append(worker)

        def _cleanup() -> None:
            if worker in self._workers:
                self._workers.remove(worker)

        def _on_error(error_text: str) -> None:
            _cleanup()
            QMessageBox.warning(self, f"{action_name}失败", error_text)
            self._set_busy(False)

        def _on_success(result: dict[str, Any]) -> None:
            _cleanup()
            on_success(result)
            self._set_busy(False)

        _ = worker.error_ready.connect(_on_error)
        _ = worker.result_ready.connect(_on_success)
        _ = worker.finished.connect(worker.deleteLater)
        worker.start()

    def _set_busy(self, busy: bool) -> None:
        self.resegment_btn.setEnabled(not busy)
        self.confirm_btn.setEnabled(not busy)
        self.status_btn.setEnabled(not busy)
        self.reset_btn.setEnabled(not busy)

    def _state_text(self) -> str:
        mapping = {
            GUIState.IDLE: "IDLE",
            GUIState.MASKS_DISPLAYED: "MASKS_DISPLAYED",
            GUIState.ASSIGNING: "ASSIGNING",
            GUIState.TRACKING: "TRACKING",
        }
        return mapping[self.state]

    def _update_status_bar(self) -> None:
        msg = (
            f"state={self._state_text()} | session_id={self.session_id} | "
            f"num_masks={self.num_masks} | assignments={len(self.assignments)}"
        )
        self.status_bar.showMessage(msg)

    def _update_assignment_display(self) -> None:
        if not self.assignments:
            self.assignment_label.setText("暂无分配")
            self._update_status_bar()
            return

        lines = []
        for mask_idx in sorted(self.assignments.keys()):
            mesh_name = os.path.basename(self.assignments[mask_idx])
            lines.append(f"mask {mask_idx} -> {mesh_name}")
        self.assignment_label.setText("\n".join(lines))
        self._update_status_bar()

    def _show_image(self, image_rgb: npt.NDArray[np.uint8]) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return

        self._latest_image = image_rgb
        h, w, _ = image_rgb.shape
        image = np.ascontiguousarray(image_rgb)
        q_img = QImage(image.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(q_img)
        scaled = pix.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _map_click_to_image(
        self, event: QMouseEvent, image_width: int, image_height: int
    ) -> tuple[int, int] | None:
        pixmap = cast(QPixmap, self.image_label.pixmap())
        if pixmap.isNull():
            return None

        px = event.position().x()
        py = event.position().y()

        label_w = self.image_label.width()
        label_h = self.image_label.height()
        pix_w = pixmap.width()
        pix_h = pixmap.height()

        offset_x = (label_w - pix_w) / 2.0
        offset_y = (label_h - pix_h) / 2.0

        if px < offset_x or py < offset_y:
            return None
        if px >= offset_x + pix_w or py >= offset_y + pix_h:
            return None

        rel_x = (px - offset_x) / float(pix_w)
        rel_y = (py - offset_y) / float(pix_h)

        img_x = int(rel_x * image_width)
        img_y = int(rel_y * image_height)
        img_x = min(max(img_x, 0), image_width - 1)
        img_y = min(max(img_y, 0), image_height - 1)
        return (img_x, img_y)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._latest_image is not None:
            self._show_image(self._latest_image)

    def on_resegment(self) -> None:
        self._set_busy(True)
        self.status_bar.showMessage("正在触发重新分割...")

        def _on_success(result: dict[str, Any]) -> None:
            if result.get("success"):
                self.session_id = int(result.get("session_id", 0))
                self.num_masks = int(result.get("num_masks", 0))
                self.assignments.clear()
                self.state = GUIState.MASKS_DISPLAYED
                self._update_assignment_display()
                self.status_bar.showMessage(
                    f"分割成功: session_id={self.session_id}, num_masks={self.num_masks}",
                    5000,
                )
            else:
                QMessageBox.warning(
                    self,
                    "Resegment失败",
                    str(result.get("message", "未知错误")),
                )
                self.state = GUIState.IDLE
                self._update_status_bar()

        self._run_worker(self.client.resegment, _on_success, "Resegment", True)

    def on_confirm(self) -> None:
        if not self.assignments:
            self.status_bar.showMessage("请先完成至少一个 mask 分配", 3000)
            return

        mask_indices = sorted(self.assignments.keys())
        mesh_paths = [self.assignments[idx] for idx in mask_indices]

        self._set_busy(True)
        self.status_bar.showMessage("正在提交模型分配...")

        def _on_success(result: dict[str, Any]) -> None:
            if result.get("success"):
                self.state = GUIState.TRACKING
                self.status_bar.showMessage("模型分配成功，已进入跟踪模式", 5000)
                self._update_status_bar()
            else:
                QMessageBox.warning(
                    self,
                    "Confirm失败",
                    str(result.get("message", "未知错误")),
                )

        self._run_worker(
            self.client.assign_models,
            _on_success,
            "Confirm",
            mesh_paths,
            mask_indices,
            self.session_id,
        )

    def on_status(self) -> None:
        self._set_busy(True)
        self.status_bar.showMessage("正在查询系统状态...")

        def _on_success(result: dict[str, Any]) -> None:
            perception = result.get("perception", {})
            controller = result.get("controller", {})
            detail = (
                f"感知状态: {perception}\n"
                f"控制器状态: {controller}\n"
                f"当前 session_id: {self.session_id}"
            )
            QMessageBox.information(self, "系统状态", detail)
            self._update_status_bar()

        self._run_worker(self.client.get_status, _on_success, "Status")

    def on_reset(self) -> None:
        self.state = GUIState.IDLE
        self.session_id = 0
        self.num_masks = 0
        self.assignments.clear()
        self._latest_image = None
        self.image_label.clear()
        self.image_label.setText("等待图像...")
        self._update_assignment_display()
        self.status_bar.showMessage("状态已重置", 3000)

    def on_image_click(self, event: QMouseEvent) -> None:
        if self.state not in (GUIState.MASKS_DISPLAYED, GUIState.ASSIGNING):
            return

        selected = cast(QListWidgetItem | None, self.mesh_list.currentItem())
        if selected is None:
            self.status_bar.showMessage("请先在右侧选择一个模型", 3000)
            return

        labels = cast(npt.NDArray[np.uint8] | None, self.client.get_masks_label())
        if labels is None:
            self.status_bar.showMessage("尚未接收到 masks_label 图像", 3000)
            return

        mapped = self._map_click_to_image(event, labels.shape[1], labels.shape[0])
        if mapped is None:
            return

        img_x, img_y = mapped
        mask_value = int(labels[img_y, img_x])
        if mask_value == 0:
            self.status_bar.showMessage("点击的是背景区域", 2000)
            return

        mask_index = mask_value - 1
        mesh_path = str(selected.data(Qt.ItemDataRole.UserRole))
        self.assignments[mask_index] = mesh_path
        self.state = GUIState.ASSIGNING
        self._update_assignment_display()
        self.status_bar.showMessage(
            f"已分配: mask {mask_index} -> {os.path.basename(mesh_path)}", 3000
        )

    def update_display(self) -> None:
        image = None
        if self.state == GUIState.TRACKING:
            image = cast(
                npt.NDArray[np.uint8] | None, self.client.get_pose_visualization()
            )
        elif self.state in (GUIState.MASKS_DISPLAYED, GUIState.ASSIGNING):
            image = cast(
                npt.NDArray[np.uint8] | None, self.client.get_masks_visualization()
            )

        if image is not None:
            self._show_image(image)

    def closeEvent(self, event) -> None:
        self.update_timer.stop()
        for worker in list(self._workers):
            worker.quit()
            _ = worker.wait(100)
        try:
            self.client.resegment(force=False)
        except Exception:
            pass
        self.client.shutdown()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="FoundationPose PySide6 GUI 客户端")
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="demo_data",
        help="模型目录（默认: demo_data）",
    )
    args = parser.parse_args()
    mesh_dir = cast(str, args.mesh_dir)

    app = QApplication(sys.argv)
    window = FoundationPoseGUI(mesh_dir=mesh_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
