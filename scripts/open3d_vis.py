import numpy as np
import open3d as o3d

class TwoCloudViewerSphere:
    def __init__(self, radius=0.02, color1=(1,0,0), color2=(0,0.8,0),
                 window_name="Two Clouds"):
        self.radius = radius
        self.color1, self.color2 = color1, color2

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name)

        self.spheres1, self.spheres2 = [], []   # Mesh 列表
        self._first_frame = True

    def _make_spheres(self, cloud, color):
        """根据首帧坐标创建球体并加入场景"""
        spheres = []
        for p in cloud:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius)
            s.translate(p)                      # 移到点的位置
            s.compute_vertex_normals()
            s.paint_uniform_color(color)
            self.vis.add_geometry(s)
            spheres.append(s)
        return spheres, cloud.copy()            # 返回球体和座标副本

    def update(self, cloud1, cloud2):
        cloud1 = cloud1.astype(np.float64)
        cloud2 = cloud2.astype(np.float64)

        if self._first_frame:
            # 建球 + 记录“上一帧”位置
            self.spheres1, self._prev1 = self._make_spheres(cloud1, self.color1)
            self.spheres2, self._prev2 = self._make_spheres(cloud2, self.color2)
            self._first_frame = False
        else:
            # 只需把每个球体平移到新位置
            for s, new_p, old_p in zip(self.spheres1, cloud1, self._prev1):
                s.translate(new_p - old_p, relative=True)
            for s, new_p, old_p in zip(self.spheres2, cloud2, self._prev2):
                s.translate(new_p - old_p, relative=True)

            # 更新“上一帧”坐标
            self._prev1[:] = cloud1
            self._prev2[:] = cloud2

            # 告诉渲染器几何体已变
            for s in self.spheres1 + self.spheres2:
                self.vis.update_geometry(s)

        self.vis.poll_events(); self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()