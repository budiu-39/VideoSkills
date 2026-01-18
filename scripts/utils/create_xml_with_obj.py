import os
import tempfile
from lxml import etree


def create_temp_xml_with_object(base_xml_path, obj_mesh_path, smpl_scale=1.0):
    """
    在现有 robot xml 中注入 object mesh 和 body。

    Args:
        base_xml_path: 机器人 XML 路径
        obj_mesh_path: 物体 Mesh 路径
        smpl_scale (float): 统一缩放比例 (应用到 x, y, z)
    """
    # 解析 XML
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(base_xml_path, parser)
    root = tree.getroot()

    # 1. 准备 Mesh 路径和名称
    if obj_mesh_path is None or not os.path.exists(obj_mesh_path):
        # 如果没有物体，直接返回原文件或不做修改
        return base_xml_path

    mesh_name = os.path.splitext(os.path.basename(obj_mesh_path))[0]
    abs_mesh_path = os.path.abspath(obj_mesh_path)

    # 构造缩放字符串 "s s s"
    scale_str = f"{smpl_scale} {smpl_scale} {smpl_scale}"

    # 2. 确保 <asset> 标签存在
    asset = root.find("asset")
    if asset is None:
        asset = etree.Element("asset")
        root.insert(0, asset)

    # 3. 注册 Mesh 资源 (Asset 去重 + 添加 Scale)
    existing_mesh = asset.find(f"mesh[@name='{mesh_name}']")

    if existing_mesh is not None:
        # 如果已存在，更新路径和缩放
        existing_mesh.set('file', abs_mesh_path)
        existing_mesh.set('scale', scale_str)
    else:
        # 创建新的 mesh asset，包含 scale 属性
        etree.SubElement(
            asset, "mesh",
            name=mesh_name,
            file=abs_mesh_path,
            scale=scale_str  # <--- 关键修改：设置缩放
        )

    # 4. 确保 <worldbody> 存在
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = etree.SubElement(root, "worldbody")

    # 5. 添加 Object Body (Body 去重)
    existing_body = worldbody.find("body[@name='object_body']")
    if existing_body is not None:
        worldbody.remove(existing_body)

    # 添加新的 body
    body_el = etree.SubElement(worldbody, "body", name="object_body", mocap="true")

    # 添加 geom 引用 mesh
    etree.SubElement(
        body_el, "geom",
        name="object_geom",
        type="mesh",
        mesh=mesh_name,
        rgba="0.2 0.8 0.2 0.6",
        contype="0",
        conaffinity="0"
    )

    # 6. 保存
    dir_name = os.path.dirname(os.path.abspath(base_xml_path))
    base_name = os.path.basename(base_xml_path)
    # 文件名加上 scale 标识防止混淆，或者直接覆盖
    output_xml_path = os.path.join(dir_name, base_name.replace(".xml", f"_{mesh_name}_scaled.xml"))

    tree.write(output_xml_path, pretty_print=True, encoding="utf-8", xml_declaration=True)

    return output_xml_path