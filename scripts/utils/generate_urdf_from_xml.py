import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
from xml.dom import minidom

# =================配置区域=================
INPUT_FILE = "/home/miku/Documents/holosoma/src/holosoma_retargeting/models/smplh/smplh_153dof.xml"  # 你的输入文件名
OUTPUT_FILE = "/home/miku/Documents/holosoma/src/holosoma_retargeting/models/smplh/smplh_153dof.urdf"  # 输出文件名
ROBOT_NAME = "smplh_153dof"


# =========================================

def vec_to_str(v):
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


def parse_vec(s):
    return np.array([float(x) for x in s.split()])


def get_rpy_from_vector(v_from, v_to):
    """
    计算从点 v_from 到 v_to 的圆柱体的中心点和旋转欧拉角(RPY)。
    URDF 默认圆柱体是沿 Z 轴竖直的。
    """
    v = v_to - v_from
    length = np.linalg.norm(v)
    center = (v_from + v_to) / 2.0

    if length < 1e-6:
        return center, np.array([0, 0, 0]), 0

    # 目标方向向量
    direction = v / length
    # 原始方向 (URDF Cylinder 默认朝 Z)
    z_axis = np.array([0, 0, 1])

    # 计算旋转
    # 旋转轴 = z_axis cross direction
    rot_axis = np.cross(z_axis, direction)
    dot_product = np.dot(z_axis, direction)

    if np.linalg.norm(rot_axis) < 1e-6:
        # 平行
        if dot_product > 0:
            r = R.identity()
        else:
            # 反向，绕X轴转180度
            r = R.from_euler('x', 180, degrees=True)
    else:
        # 计算旋转角
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        rot_vec = rot_axis / np.linalg.norm(rot_axis) * angle
        r = R.from_rotvec(rot_vec)

    rpy = r.as_euler('xyz')
    return center, rpy, length


class UrdfGenerator:
    def __init__(self, robot_name):
        self.root = ET.Element("robot", name=robot_name)
        self.materials_added = False

    def add_materials(self):
        if self.materials_added: return
        mat_skin = ET.SubElement(self.root, "material", name="skin")
        ET.SubElement(mat_skin, "color", rgba="0.8 0.6 0.4 1")
        mat_grey = ET.SubElement(self.root, "material", name="grey")
        ET.SubElement(mat_grey, "color", rgba="0.7 0.7 0.7 1")
        self.materials_added = True

    def create_link(self, name):
        link = ET.SubElement(self.root, "link", name=name)
        return link

    def add_inertial(self, link, mass=0.1):
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(mass))
        # 简单的单位惯性矩阵，避免仿真器报错
        inertia_val = str(mass * 0.01)
        ET.SubElement(inertial, "inertia",
                      ixx=inertia_val, ixy="0", ixz="0",
                      iyy=inertia_val, iyz="0",
                      izz=inertia_val)

    def add_visual_collision(self, link, geom_node):
        geom_type = geom_node.get("type", "sphere")  # default type

        # 解析几何参数
        origin_xyz = np.array([0., 0., 0.])
        origin_rpy = np.array([0., 0., 0.])
        geometry_elem = None

        # 处理 Pos 和 Quat (如果存在)
        base_pos = parse_vec(geom_node.get("pos", "0 0 0"))

        if geom_type == "capsule":
            if "fromto" in geom_node.attrib:
                ft = parse_vec(geom_node.get("fromto"))
                start, end = ft[0:3], ft[3:6]
                center, rpy, length = get_rpy_from_vector(start, end)

                origin_xyz = center
                origin_rpy = rpy
                radius = geom_node.get("size", "0.01").split()[0]  # 某些xml size有多个值

                geometry_elem = ET.Element("geometry")
                ET.SubElement(geometry_elem, "cylinder", radius=str(radius), length=str(length))
            else:
                # 标准 Capsule
                origin_xyz = base_pos
                # 假设胶囊沿Z轴
                size = geom_node.get("size", "0.01 0.05").split()
                radius = size[0]
                length = str(float(size[1]) * 2) if len(size) > 1 else "0.1"

                geometry_elem = ET.Element("geometry")
                ET.SubElement(geometry_elem, "cylinder", radius=str(radius), length=length)

        elif geom_type == "sphere":
            origin_xyz = base_pos
            radius = geom_node.get("size", "0.05")
            geometry_elem = ET.Element("geometry")
            ET.SubElement(geometry_elem, "sphere", radius=str(radius))

        elif geom_type == "box":
            origin_xyz = base_pos
            # MuJoCo size 是半长，URDF 是全长
            half_size = parse_vec(geom_node.get("size", "0.05 0.05 0.05"))
            full_size = half_size * 2.0
            geometry_elem = ET.Element("geometry")
            ET.SubElement(geometry_elem, "box", size=vec_to_str(full_size))

        # 添加 Visual
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=vec_to_str(origin_xyz), rpy=vec_to_str(origin_rpy))
        visual.append(geometry_elem)  # 重用 geometry 元素
        ET.SubElement(visual, "material", name="skin")

        # 添加 Collision (通常与 Visual 相同)
        # Deepcopy geometry for collision to strictly separate XML nodes
        import copy
        col_geom = copy.deepcopy(geometry_elem)
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=vec_to_str(origin_xyz), rpy=vec_to_str(origin_rpy))
        collision.append(col_geom)

    def process_joint(self, joint_node, parent_link, child_link, xyz_offset="0 0 0"):
        joint_name = joint_node.get("name")
        joint_type = "revolute" if joint_node.get("type") == "hinge" else "fixed"

        j_elem = ET.SubElement(self.root, "joint", name=joint_name, type=joint_type)
        ET.SubElement(j_elem, "parent", link=parent_link)
        ET.SubElement(j_elem, "child", link=child_link)
        ET.SubElement(j_elem, "origin", xyz=xyz_offset, rpy="0 0 0")

        axis = joint_node.get("axis", "0 0 1")
        ET.SubElement(j_elem, "axis", xyz=axis)

        # Limits conversion (deg -> rad)
        if "range" in joint_node.attrib:
            r = [float(x) for x in joint_node.get("range").split()]
            lower = np.radians(r[0])
            upper = np.radians(r[1])
            ET.SubElement(j_elem, "limit", lower=f"{lower:.4f}", upper=f"{upper:.4f}", effort="100", velocity="10")
        else:
            # Default limits if not specified
            ET.SubElement(j_elem, "limit", lower="-3.14", upper="3.14", effort="100", velocity="10")

    def convert_body(self, mj_body, parent_link_name):
        body_name = mj_body.get("name")
        body_pos = mj_body.get("pos", "0 0 0")

        joints = mj_body.findall("joint")
        geoms = mj_body.findall("geom")

        # 如果没有关节，这就是一个固定连接的 Link
        if not joints:
            # 这种情况下，物体实际上是父物体的一部分，或者通过固定关节连接
            # 为了保持结构清晰，我们创建一个固定关节
            link_name = body_name
            self.create_link(link_name)
            # 添加几何体
            for g in geoms:
                # 查找 link 节点并添加
                l_node = self.root.find(f".//link[@name='{link_name}']")
                self.add_visual_collision(l_node, g)
                self.add_inertial(l_node)

            # 创建 Fixed Joint
            j_name = f"{body_name}_fixed"
            j_elem = ET.SubElement(self.root, "joint", name=j_name, type="fixed")
            ET.SubElement(j_elem, "parent", link=parent_link_name)
            ET.SubElement(j_elem, "child", link=link_name)
            ET.SubElement(j_elem, "origin", xyz=body_pos, rpy="0 0 0")

            final_link_name = link_name

        else:
            # MuJoCo 这里一个 body 有多个关节 (例如 hip_x, hip_y, hip_z)
            # URDF 需要拆分成 link chain:
            # Parent -> [Joint 1] -> Virtual Link 1 -> [Joint 2] -> Virtual Link 2 -> [Joint 3] -> Real Link

            prev_link = parent_link_name
            current_offset = body_pos  # 第一个关节负责位移

            for i, joint in enumerate(joints):
                is_last_joint = (i == len(joints) - 1)

                if is_last_joint:
                    # 最后一个 Link 是实体 Link，包含几何体
                    this_link_name = body_name
                else:
                    # 中间 Link 是虚拟的
                    this_link_name = f"{body_name}_virtual_{i}"

                self.create_link(this_link_name)

                # 如果是中间虚拟 link，惯量要很小
                l_node = self.root.find(f".//link[@name='{this_link_name}']")
                self.add_inertial(l_node, mass=0.01 if not is_last_joint else 1.0)

                # 创建关节
                self.process_joint(joint, prev_link, this_link_name, current_offset)

                # 更新循环变量
                prev_link = this_link_name
                current_offset = "0 0 0"  # 后续关节原点重合

            final_link_name = body_name

            # 将几何体添加到最后一个 Link
            final_link_node = self.root.find(f".//link[@name='{final_link_name}']")
            for g in geoms:
                self.add_visual_collision(final_link_node, g)

        # 递归处理子 Body
        for child_body in mj_body.findall("body"):
            self.convert_body(child_body, final_link_name)

    def convert(self, xml_string):
        mj_tree = ET.fromstring(xml_string)
        worldbody = mj_tree.find("worldbody")

        self.add_materials()

        # 寻找根节点 (SMPL XML 中是 Pelvis)
        # MuJoCo worldbody 的直接子节点通常是 Base
        roots = worldbody.findall("body")

        for root_body in roots:
            # 根 Link (通常是 Pelvis)
            # MuJoCo 的根通常有个 freejoint，URDF 中我们把它当作 base_link
            base_name = root_body.get("name")
            base_pos = root_body.get("pos", "0 0 0")

            self.create_link(base_name)
            # 处理根节点的几何体
            base_node = self.root.find(f".//link[@name='{base_name}']")
            self.add_inertial(base_node, mass=5.0)  # 躯干质量大一点

            for g in root_body.findall("geom"):
                self.add_visual_collision(base_node, g)

            # 递归处理子节点
            for child in root_body.findall("body"):
                self.convert_body(child, base_name)

        return self.root


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


# ================= 主程序 =================
if __name__ == "__main__":
    try:
        with open(INPUT_FILE, 'r') as f:
            xml_content = f.read()

        generator = UrdfGenerator(ROBOT_NAME)
        root = generator.convert(xml_content)

        urdf_str = prettify(root)

        with open(OUTPUT_FILE, 'w') as f:
            f.write(urdf_str)

        print(f"转换成功！已生成文件: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{INPUT_FILE}'。请确保文件在当前目录下。")
    except Exception as e:
        print(f"转换过程中出错: {e}")