import os
import yaml

from videoskills.utils.torch_utils import quat_to_exp_map
from scripts.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from scripts.poselib.core.rotation3d import *
from isaacgym.torch_utils import *
from videoskills.utils.motionlib.pytorch3d_transforms import quaternion_to_matrix
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as sRot
from .. import torch_utils
import copy
from joblib import load
import torch
import joblib
import glob

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy


    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)


    torch.Tensor.numpy = Patch.numpy


class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib():
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, device):
        #
        self._rotate_motion = True
        self._fix_height = False
        #
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device

        if self._fix_height:
            self._xml_tree = ET.parse("videoskills/envs/smpl/smpl.xml")

        self._load_motions(motion_file)
        # self.preprocess_amass_motion(motion_file,)

        motions = self._motions
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(device)
        self.gas = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(device)
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(device)
        self._termination_history = torch.ones(len(self._motions), dtype=torch.float32, device=self._device)
        self._sampling_prob = torch.ones(len(self._motions), dtype=torch.float32, device=self._device)/ len(self._motions)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self._num_motions = len(self._motions)
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device).to(device)

        return

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_prob, num_samples=n, replacement=True)

        # m = self.num_motions()
        # motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        # motion_ids = torch.tensor(motion_ids, device=self._device, dtype=torch.long)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)

        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        # num_bodies = self._get_num_bodies()
        # num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]

        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        key_rot0 = self.grs[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_rot1 = self.grs[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0,
                key_pos1, key_rot0, key_rot1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)
        blend_exp = blend.unsqueeze(1)
        key_rot = torch_utils.slerp(key_rot0, key_rot1, blend_exp)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1

        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        key_vel0 = self.gvs[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_vel1 = self.gvs[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_vel = (1.0 - blend_exp) * key_vel0 + blend_exp * key_vel1

        key_ang_vel0 = self.gas[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_ang_vel1 = self.gas[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_ang_vel = (1.0 - blend_exp) * key_ang_vel0 + blend_exp * key_ang_vel1

        motion_state = {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel": dof_vel,
            "key_pos": key_pos,
            "key_rot": key_rot,
            "key_vel": key_vel,
            "key_ang_vel": key_ang_vel,
        }

        return motion_state

    def _load_motions(self, motion_file, skeleton_trees = None):
        # TODO: Add support for offset
        self._motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_keys = []

        total_len = 0.0

        motion_files = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion = SkeletonMotion.from_file(curr_file)

            # trans, trans_fix = self.fix_trans_height(pose_aa, trans, 0, mesh_parsers)

            # sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees[f], pose_quat_global, trans,
            #                                                             is_local=False)

            # curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))

            if self._rotate_motion:
                curr_motion = self.apply_rotation(curr_motion, curr_motion.fps)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_keys.append(self._get_amass_key(curr_file))
            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)

            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)
            # else:
            #     curr_motion.tensor = curr_motion.tensor.to(self._device)
            #     curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
            #     curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(
            #         self._device)
            #     curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

            self._motion_files.append(curr_file)

        self._sort_motions_by_length()

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)


        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        num_motions = len(self._motions)
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _calc_frame_blend(self, time, len, num_frames, dt):

        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)

        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels

    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1]  # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert (False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert (joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1]  # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert (False)

        return dof_vel

    def apply_rotation(self, motion, fps, rotation_angle=None):
        """
        Apply a rotation to the motion data.

        Args:
            motion (SkeletonMotion): The motion to rotate.
            rotation_angle (float, optional): The angle to rotate (in radians). If None, a random angle is sampled.
        """
        # Sample a random angle if rotation_angle is not provided
        if rotation_angle is None:
            rotation_angle = torch.rand(1) * 2 * torch.pi  # Random angle in [0, 2π]

        # Create a rotation quaternion for the z-axis
        rotation_quat = torch_utils.quat_from_angle_axis(rotation_angle,
                                                         torch.tensor([0.0, 0.0, 1.0]))

        global_translation = torch_utils.quat_apply(rotation_quat, motion.global_translation[:,0])

        if rotation_quat.shape != motion.global_rotation.shape:
            rotation_quat = rotation_quat.expand_as(motion.global_rotation)

        # Apply the rotation to the global rotation
        global_rotation = torch_utils.quat_mul(rotation_quat, motion.global_rotation)

        # Apply the rotation to the global translation

        # Create a new SkeletonState with the rotated global rotation and translation
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            motion.skeleton_tree,
            global_rotation,
            global_translation,  # Ensure translation is 3D
            is_local=False
        )

        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)

        return new_motion

    def fix_motion_heights(motion, skeleton_tree):
        if skeleton_tree is None:
            if hasattr(motion, "skeleton_tree"):
                skeleton_tree = motion.skeleton_tree
        body_heights = motion.global_translation[..., 2]
        min_height = body_heights.min()

        if skeleton_tree is None:
            motion.global_translation[..., 2] -= min_height
            return motion

        root_translation = motion.root_translation
        root_translation[:, 2] -= min_height

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            motion.global_rotation,
            root_translation,
            is_local=False,
        )

        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

        return new_motion

    def _sort_motions_by_length(self):
        sorted_indices = torch.argsort(torch.tensor(self._motion_lengths))  # on CPU first

        # 如果你已经有 tensor，就用它；否则用原始 list 排序
        self._motions = [self._motions[i] for i in sorted_indices]
        self._motion_files = [self._motion_files[i] for i in sorted_indices]
        self._motion_lengths = [self._motion_lengths[i] for i in sorted_indices]
        self._motion_fps = [self._motion_fps[i] for i in sorted_indices]
        self._motion_dt = [self._motion_dt[i] for i in sorted_indices]
        self._motion_num_frames = [self._motion_num_frames[i] for i in sorted_indices]
        self._motion_keys = [self._motion_keys[i] for i in sorted_indices]

    def _get_amass_key(self, filepath):
        """
        Generate a unique key for each motion file in the format: SUBSET-SUBFOLDER-FILENAME
        Example: 'CMU-86_05-walk_01'
        """
        import os
        parts = filepath.split(os.sep)
        if len(parts) >= 4:
            subset = parts[-3]
            subfolder = parts[-2]
            filename = os.path.splitext(parts[-1])[0]
            return f"{subset}-{subfolder}-{filename}"
        else:
            return "UNKNOWN-UNKNOWN-UNKNOWN"

    def update_soft_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only "mostly" trained on "failed" sequences. Auto PMCP.
        if len(failed_keys) > 0:
            all_keys = self._motion_keys
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self._sampling_prob[:] = self._termination_history / self._termination_history.sum()

    def export_sampling_state(self, filepath):
        """
        Export the current termination history, sampling probability, and motion keys to a .pkl file.
        The keys in the dictionary are motion keys, and each entry is a dict with:
            - 'termination_count'
            - 'sampling_prob'
        """
        assert hasattr(self, "_termination_history") and hasattr(self, "_sampling_prob"), \
            "Termination history or sampling probabilities not found"

        export_dict = {}
        for i, key in enumerate(self._motion_keys):
            export_dict[key] = {
                "termination_count": self._termination_history[i].item(),
                "sampling_prob": self._sampling_prob[i].item(),
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(export_dict, filepath, compress=True)
        print(f"[MotionLib] Sampling state exported to {filepath}")

    def _fix_height_based_on_geom(self, curr_motion):
        return



    def _fetch_motion_files(self, input_motion_sequences: str, ext=".npy", amass_root="AMASS_processed") -> list:
        """
        Load all motion file paths.
        - If input is a folder: recursively find all files ending with `ext`.
        - If input is a .pkl file: load failed_keys and convert to full file paths.

        Args:
            input_motion_sequences (str): directory or .pkl file
            ext (str): file extension to match (default: '.npy')
            amass_root (str): root directory to prefix when generating paths from keys

        Returns:
            List[str]: list of full file paths
        """

        if isinstance(input_motion_sequences, list):
            return input_motion_sequences
        else:
            motion_paths = glob.glob(os.path.join(input_motion_sequences, f"**/*{ext}"), recursive=True)
            motion_paths.sort()
            return motion_paths


