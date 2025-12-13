import numpy as np
from scripts.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
from scripts.utils.smpl_humanoid_tool import humanoid2smpl
from scripts.rep_272.smpl_to_d272 import smpl_to_272d, yup2zup
from scripts.utils.body_model_smplx import BodyModelSMPLX

fpath = '/home/miku/Documents/VideoSkills/dataset/smpl_motion/AMASS_train_fixed_height/BioMotionLab_NTroje/rub018/0014_knocking2_poses.npy'

motion_dict_a = np.load(fpath, allow_pickle=True).item()
sk_a = SkeletonState.from_dict(motion_dict_a)

position_data = sk_a.global_translation[:,0]
pred_rot = sk_a.global_rotation

smpl_parser_n = BodyModelSMPLX(model_path='data/SMPL',
                               model_type='smplx')
skeleton_tree = SkeletonTree.from_mjcf(f"data/robots/smpl/smpl_humanoid.xml")
pose_aa, transl = humanoid2smpl(pred_rot, position_data, skeleton_tree, is_smplh=False)
nfrm = pose_aa.shape[0]
smpl_parser_n.cuda()
smpl_parser_n.eval()
beta = np.zeros((nfrm, 10), dtype=np.float32)

transl, pose_aa[:, :3] = yup2zup(transl, pose_aa[:, :3])

smpl_data = np.concatenate([pose_aa.reshape(nfrm, -1)[..., :66], np.zeros((transl.shape[0], 6)),
                            transl, np.zeros((transl.shape[0], 10))], axis=-1)

x272 = smpl_to_272d(smpl_data[:, 72:75], smpl_data[:, :72].reshape(nfrm, -1, 3), smpl_data[:, 75:], smpl_parser_n)


root_272 = '/home/miku/Documents/VideoSkills/dataset/272_rep/AMASS_272'
fpath_272 = f'{root_272}/BioMotionLab_NTroje-rub018-0014_knocking2_poses.npy'
motion_272 = np.load(fpath_272, allow_pickle=True)

print(sk_a.root_translation.shape)
print(motion_dict_a.shape)
print("done")

