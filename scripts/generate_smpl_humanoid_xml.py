import torch
import numpy as np
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

def set_joint_ds(robot, default_d=50.0, default_k=500.0, per_joint=None):
    """per_joint: {base_name: (damping, stiffness)}; e.g. 'L_Hip' 会作用于 L_Hip_x/y/z"""
    hits = 0
    for body in robot.bodies:
        for j in body.joints:
            if j.type != "hinge":
                continue
            d, k = default_d, default_k
            if per_joint:
                for base, (d0, k0) in per_joint.items():
                    if j.name == base or j.name.startswith(base + "_"):
                        d, k = d0, k0
                        hits += 1
                        break
            j.damping   = np.array([float(d)])
            j.stiffness = np.array([float(k)])
    robot.sync_node()
    # 调试：看看命中的关节数量
    # print(f"matched {hits} hinge joints")

robot_cfg = {
    "model": "smpl",   # 若要 SMPL-H，请改成 "smplh"
    "mesh": False,
    "upright_start": True,
    "freeze_hand": True,
    "replace_feet": True,
    "real_weight": True,
    "real_weight_porpotion_capsules": True,
    "big_ankle": True,
    "box_body": False,
    "masterfoot": False,
    "body_params": {}, "joint_params": {}, "geom_params": {}, "actuator_params": {},
}

smpl_robot = SMPL_Robot(robot_cfg, data_dir="data/SMPL/smpl")

# 1) 先构模
smpl_robot.load_from_skeleton(betas=torch.zeros(1, 10), gender=[1])  # 0=neutral, 1=male, 2=female

# 2) 再设置各关节的阻尼/刚度
joint_names_800  = ['L_Hip','L_Knee','L_Ankle','R_Hip','R_Knee','R_Ankle']
joint_names_500  = ['L_Toe','R_Toe','Neck','Head','L_Thorax','L_Shoulder','L_Elbow','R_Thorax','R_Shoulder','R_Elbow']
joint_names_1000 = ['Torso','Spine','Chest']
joint_names_300  = ['L_Wrist','L_Hand','R_Wrist','R_Hand']

per_joint = {}
per_joint.update({n: (80,  800)  for n in joint_names_800})
per_joint.update({n: (50,  500)  for n in joint_names_500})
per_joint.update({n: (100, 1000) for n in joint_names_1000})
per_joint.update({n: (30,  300)  for n in joint_names_300})

# 默认兜底值：没列出的关节就用它（例如 50/500）
set_joint_ds(smpl_robot, default_d=10.0, default_k=100.0, per_joint=per_joint)

# 3) 写 XML
smpl_robot.write_xml("/home/miku/Documents/VideoSkills/data/robots/smplh/smplh_humanoid_v1.xml")

