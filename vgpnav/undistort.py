"""鱼眼 -> 针孔 去畸变。支持两种相机模型:

  · fisheye (KB4 / Kannala-Brandt, OpenCV fisheye): 内参 efl_x,efl_y,cod_x,cod_y + k1..k4
        取自 camera_param.json (ChuangfuTower 头盔)。去畸变用 cv2.fisheye.initUndistortRectifyMap。
  · mei (全向相机, MEI): 内参 [xi,fx,fy,cx,cy] + 畸变 [k1,k2,p1,p2]
        取自 memory-nav/cam/params.yaml。投影逐式移植自 C++ projectPoints。

统一接口: load_camera_params(path, cam_name) -> params(dict, 含 'model');
PinholeUndistorter(params, ...) 输出针孔图 (VGGT 假设针孔输入)。
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import yaml


class PinholeUndistorter:
    """把单台鱼眼/全向相机去畸变为针孔图像, 并给出针孔内参 K。

    Args:
        params: load_camera_params 返回的 dict (含 'model' 与内参)
        out_w, out_h: 输出针孔图尺寸
        hfov_deg: 输出针孔图水平视场角 (度), 决定虚拟焦距
        pitch_down_deg: 虚拟相机向下俯仰角 (度), 正数向下看更多地面
    """

    def __init__(self, params, out_w=640, out_h=480, hfov_deg=110.0, pitch_down_deg=0.0):
        self.p = params
        self.model = params["model"]
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

    def _project_mei(self, X: np.ndarray):
        """X: (...,3) 相机系 3D -> MEI 鱼眼像素 (u,v)。逐式对齐 C++ projectPoints。"""
        p = self.p
        xs = X / np.linalg.norm(X, axis=-1, keepdims=True)
        denom = xs[..., 2] + p["xi"]
        xu = xs[..., 0] / denom
        yu = xs[..., 1] / denom
        r2 = xu * xu + yu * yu
        radial = 1.0 + p["k"][0] * r2 + p["k"][1] * r2 * r2
        xd = xu * radial + 2 * p["k"][2] * xu * yu + p["k"][3] * (r2 + 2 * xu * xu)
        yd = yu * radial + p["k"][2] * (r2 + 2 * yu * yu) + 2 * p["k"][3] * xu * yu
        return p["fx"] * xd + p["cx"], p["fy"] * yd + p["cy"]

    def _build_maps(self):
        # 虚拟相机俯仰 (绕 x 轴; 相机系 y 向下, 正 pitch_down 看更多地面)
        th = np.radians(self.pitch_down_deg)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
        if self.model == "fisheye":
            # KB4: cv2.fisheye.initUndistortRectifyMap(K_fish, D, R, K_pinhole, size)
            p = self.p
            K = np.array([[p["efl_x"], 0, p["cod_x"]],
                          [0, p["efl_y"], p["cod_y"]], [0, 0, 1]], dtype=np.float64)
            D = np.array(p["k"], dtype=np.float64).reshape(4, 1)  # k1..k4
            self.map_x, self.map_y = cv2.fisheye.initUndistortRectifyMap(
                K, D, R, self.K, (self.out_w, self.out_h), cv2.CV_32FC1)
        else:
            # MEI: 手写逆映射 (针孔射线 -> _project_mei -> 鱼眼像素)
            xs = np.arange(self.out_w) - self.cx_v
            ys = np.arange(self.out_h) - self.cy_v
            gx, gy = np.meshgrid(xs, ys)
            dirs = np.stack([gx, gy, np.full_like(gx, self.f_v)], axis=-1) @ R.T
            u, v = self._project_mei(dirs)
            self.map_x = u.astype(np.float32)
            self.map_y = v.astype(np.float32)

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)


def load_camera_params(path: str, cam_name: str) -> dict:
    """读取指定相机参数, 返回 dict(含 'model')。

    .json -> camera_param.json (KB4 fisheye / pinhole);  .yaml -> params.yaml (MEI)。
    """
    if path.endswith(".json"):
        txt = open(path).read()
        txt = txt[txt.index("{"):]              # 跳过可能的终端输出污染首行
        d = json.loads(txt)
        for c in d["cameras"]:
            if c.get("name") == cam_name:
                dc = c["distortion_coefficients"]
                if c["model"] == "fisheye":
                    return {"model": "fisheye", "efl_x": c["efl_x"], "efl_y": c["efl_y"],
                            "cod_x": c["cod_x"], "cod_y": c["cod_y"],
                            "k": [dc["k1"], dc["k2"], dc["k3"], dc["k4"]]}
                return {"model": "pinhole", "efl_x": c["efl_x"], "efl_y": c["efl_y"],
                        "cod_x": c["cod_x"], "cod_y": c["cod_y"]}
        raise KeyError(f"相机 {cam_name} 不在 {path} 中")
    # YAML MEI
    cfg = yaml.safe_load(open(path))
    for c in cfg["cameras"]:
        if c.get("name") == cam_name:
            xi, fx, fy, cx, cy = c["intrinsics"]
            k1, k2, p1, p2 = c["distortion_coeffs"]
            return {"model": "mei", "xi": xi, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "k": [k1, k2, p1, p2]}
    raise KeyError(f"相机 {cam_name} 不在 {path} 中")
