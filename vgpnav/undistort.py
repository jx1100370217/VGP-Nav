"""鱼眼 (MEI / 全向相机模型) -> 针孔 去畸变。

相机内参取自 memory-nav/cam/params.yaml:
    intrinsics = [xi, fx, fy, cx, cy]
    distortion_coeffs = [k1, k2, p1, p2]

投影公式 (3D 相机坐标 -> 鱼眼像素) 移植自
memory-nav/cam/tools/fisheye_undist_cpu.h 的 projectPoints, 与原始 C++
实现逐式对齐; 区别在于这里把输出从柱面 (cylindrical) 改为针孔
(perspective), 因为 VGGT 假设输入是针孔图像。
"""
from __future__ import annotations

import cv2
import numpy as np
import yaml


class PinholeUndistorter:
    """把单台鱼眼相机去畸变为针孔图像, 并给出对应的针孔内参 K。

    Args:
        intrinsics: [xi, fx, fy, cx, cy]
        dist_coeffs: [k1, k2, p1, p2]
        out_w, out_h: 输出针孔图尺寸
        hfov_deg: 输出针孔图水平视场角 (度), 决定虚拟焦距
        pitch_down_deg: 虚拟相机向下俯仰角 (度), 正数向下看更多地面
    """

    def __init__(self, intrinsics, dist_coeffs, out_w=640, out_h=480,
                 hfov_deg=110.0, pitch_down_deg=0.0):
        self.xi, self.fx, self.fy, self.cx, self.cy = [float(v) for v in intrinsics]
        self.k1, self.k2, self.p1, self.p2 = [float(v) for v in dist_coeffs]
        self.out_w, self.out_h = int(out_w), int(out_h)
        self.f_v = (self.out_w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.cx_v, self.cy_v = self.out_w / 2.0, self.out_h / 2.0
        self.pitch_down_deg = float(pitch_down_deg)
        self._build_maps()

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f_v, 0, self.cx_v],
                         [0, self.f_v, self.cy_v],
                         [0, 0, 1]], dtype=np.float64)

    def _project(self, X: np.ndarray):
        """X: (...,3) 相机系 3D -> 鱼眼像素 (u, v)。逐式对齐 C++ projectPoints。"""
        xs = X / np.linalg.norm(X, axis=-1, keepdims=True)
        denom = xs[..., 2] + self.xi
        xu = xs[..., 0] / denom
        yu = xs[..., 1] / denom
        r2 = xu * xu + yu * yu
        r4 = r2 * r2
        radial = 1.0 + self.k1 * r2 + self.k2 * r4
        xd = xu * radial + 2 * self.p1 * xu * yu + self.p2 * (r2 + 2 * xu * xu)
        yd = yu * radial + self.p1 * (r2 + 2 * yu * yu) + 2 * self.p2 * xu * yu
        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        return u, v

    def _build_maps(self):
        xs = np.arange(self.out_w) - self.cx_v
        ys = np.arange(self.out_h) - self.cy_v
        gx, gy = np.meshgrid(xs, ys)  # (H, W)
        # 针孔射线: 直接线性 (无柱面 tan 变换)
        dirs = np.stack([gx, gy, np.full_like(gx, self.f_v)], axis=-1)  # (H,W,3)
        # 可选向下俯仰 (绕相机 x 轴; 相机系 y 向下, 故正 pitch_down 让 z 朝 +y)
        th = np.radians(self.pitch_down_deg)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # R_x(th)
        dirs = dirs @ R.T
        u, v = self._project(dirs)
        self.map_x = u.astype(np.float32)
        self.map_y = v.astype(np.float32)

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)


def load_camera_params(yaml_path: str, cam_name: str):
    """从 cam/params.yaml 读取指定相机的 (intrinsics, distortion_coeffs)。"""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    for c in cfg["cameras"]:
        if c.get("name") == cam_name:
            return c["intrinsics"], c["distortion_coeffs"]
    raise KeyError(f"相机 {cam_name} 不在 {yaml_path} 中")
