#!/usr/bin/env python
# coding: utf-8

# In[30]:


import traceback
from os.path import join as pjoin

from ...humanml.common.skeleton import Skeleton
import numpy as np
import os
from ...humanml.common.quaternion import *
# from paramUtil import *

import torch
from tqdm import tqdm, trange
import os
from pathlib import Path

# In[31]:


FPS = 20
NYMERIA_DATA_PATH = Path("/local/shared_data/nymeria")
# NYMERIA_DATA_PATH = Path("/media/dominik/DominikSSD/AriaData/data/all_data/")

# we only use a subset of them

# In[32]:


momentum_used_joints = [
    # (0, 'body_world'),
    (1, 'b_root'),
    (2, 'b_spine0'),
    (3, 'b_spine1'),
    (4, 'b_spine2'),
    (5, 'b_spine3'),
    (6, 'b_neck0'),
    (7, 'b_head'),
    (8, 'b_head_null'),
    # (37, 'p_neck_twist0'),
    # (38, 'p_neck_twist1'),
    (39, 'b_r_shoulder'),
    # (40, 'p_r_delt'),
    (41, 'p_r_scap'),
    (42, 'b_r_arm'),
    # (43, 'p_r_arm_twist4'),
    # (44, 'p_r_arm_twist3'),
    # (45, 'p_r_arm_twist2'),
    # (46, 'p_r_arm_twist1'),
    # (47, 'p_r_arm_twist0'),
    (48, 'b_r_forearm'),
    # (49, 'p_r_forearm_twist0'),
    # (50, 'p_r_forearm_twist1'),
    # (51, 'p_r_forearm_twist2'),
    # (52, 'p_r_forearm_twist3'),
    # (53, 'p_r_forearm_twist4'),
    (54, 'b_r_wrist_twist'),
    (55, 'b_r_wrist'),
    (77, 'b_l_shoulder'),
    # (78, 'p_l_delt'),
    (79, 'p_l_scap'),
    (80, 'b_l_arm'),
    # (81, 'p_l_arm_twist4'),
    # (82, 'p_l_arm_twist3'),
    # (83, 'p_l_arm_twist2'),
    # (84, 'p_l_arm_twist1'),
    # (85, 'p_l_arm_twist0'),
    (86, 'b_l_forearm'),
    # (87, 'p_l_forearm_twist0'),
    # (88, 'p_l_forearm_twist1'),
    # (89, 'p_l_forearm_twist2'),
    # (90, 'p_l_forearm_twist3'),
    # (91, 'p_l_forearm_twist4'),
    (92, 'b_l_wrist_twist'),
    (93, 'b_l_wrist'),
    (115, 'p_sternum'),
    (116, 'p_navel'),
    (117, 'b_r_upleg'),
    (118, 'b_r_leg'),
    # (119, 'p_r_leg_twist0'),
    # (120, 'p_r_leg_twist1'),
    # (121, 'p_r_leg_twist2'),
    # (122, 'p_r_leg_twist3'),
    # (123, 'p_r_leg_twist4'),
    # (124, 'b_r_foot_twist'), # not used due to 0 offset
    # (125, 'b_r_foot'),
    (126, 'b_r_talocrural'),
    (127, 'b_r_subtalar'),
    (128, 'b_r_transversetarsal'),
    (129, 'b_r_ball'),
    # (130, 'p_r_upleg_twist0'),
    # (131, 'p_r_upleg_twist1'),
    # (132, 'p_r_upleg_twist2'),
    # (133, 'p_r_upleg_twist3'),
    # (134, 'p_r_upleg_twist4'),
    (135, 'b_l_upleg'),
    (136, 'b_l_leg'),
    # (137, 'p_l_leg_twist0'),
    # (138, 'p_l_leg_twist1'),
    # (139, 'p_l_leg_twist2'),
    # (140, 'p_l_leg_twist3'),
    # (141, 'p_l_leg_twist4'),
    # (142, 'b_l_foot_twist'), # not used due to 0 offset
    # (143, 'b_l_foot'),
    (144, 'b_l_talocrural'),
    (145, 'b_l_subtalar'),
    (146, 'b_l_transversetarsal'),
    (147, 'b_l_ball'),
    # (148, 'p_l_upleg_twist0'),
    # (149, 'p_l_upleg_twist1'),
    # (150, 'p_l_upleg_twist2'),
    # (151, 'p_l_upleg_twist3'),
    # (152, 'p_l_upleg_twist4'),
    # (153, 'p_pelvis'),
    # (154, 'p_pelvis_null'),
    # (155, 'p_r_rect'),
    # (156, 'p_l_glut'),
    # (157, 'p_r_glut'),
    # (158, 'p_l_rect'),
]
print(f"Using {len(momentum_used_joints)} joints for momentum representation.")

# In[62]:


# default offsets in y up and cm
tgt_offsets_cm = np.array(
    [[0.0000, 0.0000, 0.0000], [0.0000, 2.0220, -3.2536], [0.0000, 11.0412, 1.1038], [0.0000, 10.9368, -2.5953],
     [0.0000, 18.4331, 1.0320], [0.0000, 12.0983, 5.2203], [0.0000, 6.7689, 2.0376], [0.0000, 20.3080, 0.0000],
     [-2.8218, 5.7188, 10.0692], [-13.2272, 4.3198, -9.5595], [-1.5371, -2.9307, -0.0127], [-17.2036, -19.0165, 1.3635],
     [-17.0835, -10.6669, 11.5681], [-1.1367, -0.3358, 2.7043], [2.8218, 5.7193, 10.0692], [13.2272, 4.3190, -9.5595],
     [1.5371, -2.9300, -0.0127],
     [17.2036, -19.0170, 1.3635], [17.0835, -10.6670, 11.5681], [1.1367, -0.3350, 2.7043], [0.0000, 3.0884, 15.5552],
     [0.0000, -12.0104, -0.0212],
     [-7.9869, -2.5499, -0.5283], [-5.7317, -41.5127, -2.4222], [-5.5137, -41.4732, -4.2988],
     [-0.9378, -3.3756, 0.5377], [-0.0486, 1.0518, 6.0182],
     [-0.5709, -2.9564, 7.1731], [7.9869, -2.5499, -0.5283], [5.7317, -41.5127, -2.4222], [5.5137, -41.4732, -4.2989],
     [0.9378, -3.3756, 0.5377],
     [0.0486, 1.0518, 6.0182], [0.5709, -2.9564, 7.1731]]
    , dtype=np.float32)
tgt_offsets = torch.from_numpy(tgt_offsets_cm) / 100.0  # convert to meters

# make unit vectors

norm = torch.norm(tgt_offsets, dim=1, keepdim=True)
norm[0] = 1.0  # keep root at (0,0,0)
n_raw_offsets = tgt_offsets / norm
# n_raw_offsets = torch.from_numpy(n_raw_offsets)

kinematic_chain = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13],
                   [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19], [0, 1, 2, 3, 20, 21], [0, 22, 23, 24, 25, 26, 27],
                   [0, 28, 29, 30, 31, 32, 33]]

# from loguru import logger
# import sys
#
# logger.remove(0)
# logger.add(sys.stderr, level="WARNING")
# This would disable all logs originating from the "nymeria" module.
# logger.disable("MultiRecordFileReader")
# logger.disable("VrsDataProvider")
# logger.disable("MpsDataPathsProvider")

momentum_used_joint_idx = [joint[0] for joint in momentum_used_joints]


def save_joints(momentum_motion_path: Path, fps_target: int = FPS):
    """Saves all the used joints from the nymeria file path at FPS.

    The joints are saved in y-up coordinate system, in meters.

    Args:
        momentum_motion_path (Path): The path to the momentum motion file.
    """
    nymeria_data_provider = NymeriaDataProvider(sequence_rootdir=momentum_motion_path, load_wrist=True)
    logger.remove()  # Remove the default stderr sink.

    t_ns_start, t_ns_end = nymeria_data_provider.timespan_ns
    dt: int = int(1e9 / fps_target)

    joint_positions = []
    for idx, t_ns in enumerate(trange(t_ns_start, t_ns_end, dt)):
        poses: dict[str, any] = nymeria_data_provider.get_synced_poses(t_ns)
        momentum_joints = poses["joints_momentum"]
        used_joint_positions_m = momentum_joints[momentum_used_joint_idx, :3]
        joint_positions.append(used_joint_positions_m.cpu().numpy())

    joint_positions_m = np.stack(joint_positions, axis=0)

    output_path = momentum_motion_path / "body/joint_positions.npy"

    if np.isnan(joint_positions_m).any():
        logger.error(f"NaN values found in joint positions for {momentum_motion_path}.")
        raise ValueError("NaN values found in joint positions.")
    np.save(output_path, joint_positions_m)
    print(f"Saved joint positions to {output_path}, shape: {joint_positions_m.shape}")


# for momentum_motion_path in tqdm(all_nymeria_data_paths, desc="Processing momentum motion files"):
#     try:
#         save_joints(momentum_motion_path)
#     except Exception as e:
#         print(f"Error processing {momentum_motion_path}: {e}")
#         # print stack trace
#         traceback.print_exc()


# In[71]:


import numpy as np

# 1. Turn floating‐point errors into exceptions to catch divide/invalid ops at source
np.seterr(divide='raise', invalid='raise')


def uniform_skeleton(positions, target_offset):
    """
    Retargets a motion sequence to a skeleton with standardized bone lengths.

    This function normalizes the skeleton's proportions by scaling it to match
    a target skeleton's bone lengths. It preserves the original motion's rotational
    data (pose) and root trajectory, applying them to the new standardized skeleton.

    The scaling factor is determined by comparing the leg lengths of the source
    and target skeletons.

    Args:
        positions (np.ndarray): The input motion sequence, as a numpy array of
            joint positions with shape (seq_len, joints_num, 3).
        target_offset (torch.Tensor): The desired bone offsets for the target
            skeleton, with shape (joints_num, 3).

    Returns:
        np.ndarray: A new motion sequence with the same shape as the input, but
            with the joint positions adjusted to the target skeleton's proportions.
    """
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


# In[60]:

def fill_nan_with_previous_frame(positions):
    """
    Replaces frames containing NaN values with the last valid frame.

    Args:
        positions (np.ndarray): A 3D NumPy array of shape (frames, joints, coordinates)
                                that may contain NaN values.

    Returns:
        np.ndarray: A new NumPy array with NaN frames replaced.
    """
    # Create a copy to avoid modifying the original array
    filled_positions = np.copy(positions)

    last_valid_frame = None

    for i in range(len(filled_positions)):
        # Check if the current frame contains any NaNs across the joint and coordinate axes
        if np.any(np.isnan(filled_positions[i])):
            # If the first frame is NaN, we cannot fill it yet.
            # This check ensures we have a valid frame to copy from.
            if last_valid_frame is not None:
                # Replace the entire NaN frame with the last valid frame
                filled_positions[i] = last_valid_frame
        else:
            # If the current frame is valid, update the last_valid_frame
            last_valid_frame = filled_positions[i]

    # After the first pass, it's possible the very first frames were NaN and were not filled.
    # We can handle this by back-filling from the first valid frame.
    # Find the first valid frame index
    first_valid_frame_idx = np.where(~np.any(np.isnan(filled_positions), axis=(1, 2)))[0]

    if len(first_valid_frame_idx) > 0:
        first_valid_idx = first_valid_frame_idx[0]
        # Fill any leading NaN frames with the first valid frame
        for i in range(first_valid_idx):
            filled_positions[i] = filled_positions[first_valid_idx]

    return filled_positions


def process_file(positions_orig, feet_thre):
    """
    Processes a single motion sequence to extract a comprehensive feature representation.

    This function takes raw joint positions, normalizes them, and computes a rich
    feature vector for each frame, including root and joint kinematics, and foot contacts.

    Args:
        positions (np.ndarray): A numpy array of motion data with shape (seq_len, joints_num, 3).
        feet_thre (float): The threshold for foot contact detection.

    Returns:
        tuple: A tuple containing:
            - data (np.ndarray): The processed feature vector of shape (seq_len - 1, 263).
              This vector is a concatenation of the following features for each frame:
                - Root Information (4 dimensions, indices `[:4]`):
                    [0]: Root's angular velocity around the y-axis.
                    [1:3]: Root's linear velocity on the xy-plane.
                    [3]: Root's z-velocity (height velocity).
                - Rotation-Invariant Joint Positions (63 dimensions, indices `[4:67]`):
                    Local positions of the 21 joints (excluding the root) relative
                    to the root, resulting in (21 * 3) dimensions.
                - Joint Rotations (126 dimensions, indices `[67:193]`):
                    Continuous 6D rotation representation for the 21 joints
                    (excluding the root), resulting in (21 * 6) dimensions.
                - Local Joint Velocities (66 dimensions, indices `[193:259]`):
                    Velocities of all 22 joints in the root's local coordinate system,
                    resulting in (22 * 3) dimensions.
                - Foot Contacts (4 dimensions, indices `[259:263]`):
                    Binary contact flags for the left and right feet (2 per foot).
            - global_positions (np.ndarray): Ground truth positions after normalization,
              shape (seq_len, joints_num, 3).
            - positions (np.ndarray): Local, root-relative joint positions,
              shape (seq_len, joints_num, 3).
            - l_velocity (np.ndarray): The root's linear velocity on the xy-plane,
              shape (seq_len - 1, 2).
    """
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions_orig, tgt_offsets)
    # rerun_log_joint_animation([positions, positions_orig], kinematic_chain)

    positions = fill_nan_with_previous_frame(positions)

    '''Put on Floor, up is z'''
    floor_height = positions.min(axis=0).min(axis=0)[2]
    positions[:, :, 2] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''Xy at origin'''
    root_pos_init = positions[0]
    root_pose_init_xy = root_pos_init[0] * np.array([1, 1, 0])
    positions = positions - root_pose_init_xy

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face y+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around z-axis
    forward_init = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 1, 0]])  # Target forward direction (y+)
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)
    # rerun_log_joint_animation([positions], kinematic_chain)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions, r_rot):
        """
        Calculates the Rotation-Invariant Forward Kinematics Embedding.

        This function transforms global joint positions into a local,
        rotation-invariant representation. The resulting positions describe the
        pose of the skeleton relative to the root joint's coordinate system for
        each frame. This makes the representation independent of the character's
        global orientation.

        The process involves two steps:
        1.  Translating the skeleton so the root is at the origin.
        2.  Rotating the skeleton to align with the root's local coordinate system.

        Args:
            positions (np.ndarray): A numpy array of global joint positions with
                shape (sequence_length, num_joints, 3).

        Returns:
            np.ndarray: A numpy array of the same shape containing the joint
                positions in the root's local coordinate system.
        """
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 1] -= positions[:, 0:1, 1]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Y+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        """
                Computes motion parameters including 6D rotations and root velocities.

        This function processes a sequence of global joint positions to extract
        key kinematic features. It performs inverse kinematics to get joint
        rotations as quaternions, which are then converted to a continuous 6D
        representation suitable for neural networks. It also calculates the
        root's linear and angular velocities for each frame.

        Args:
            positions (np.ndarray): A numpy array of global joint positions with
                shape (seq_len, joints_num, 3).

        Returns:
            tuple: A tuple containing:
                - cont_6d_params (np.ndarray): The continuous 6D rotation for
                  all 22 joints, shape (seq_len, joints_num, 6). The root
                  joint's rotation is in the global frame, while all other
                  joint rotations are in the local kinematic chain frame
                  (relative to their parent).
                - r_velocity (np.ndarray): The root's angular velocity between
                  frames, represented as a quaternion, shape (seq_len - 1, 4).
                  This represents the change in global orientation.
                - velocity (np.ndarray): The root's linear velocity in its
                  local coordinate system, shape (seq_len - 1, 3).
                - r_rot (np.ndarray): The root's global orientation as a
                  quaternion for each frame, shape (seq_len, 4).
        """
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        # run fk
        # skel.set_offset(tgt_offsets)
        # pos_fk = skel.forward_kinematics_np(quat_params, positions[:, 0])

        # rerun_log_joint_animation([positions,pos_fk], kinematic_chain)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)

    # rerun_log_root_rotation(r_rot, np.zeros_like(positions))

    # rerun_log_joint_animation([positions], kinematic_chain)

    positions = get_rifke(positions, r_rot)

    # rerun_log_joint_animation([positions], kinsematic_chain)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_z = positions[:, 0, 2:3]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along z-axis
    # (seq_len-1, 2) linear velocity
    # r_velocity: Root's angular velocity around the z-axis (in radians).
    # r_velocity = theta/2
    r_velocity = np.arcsin(r_velocity[:, 3:4])
    # l_velocity: Root's linear velocity on the xy-plane, in the root's local coordinate system.
    l_velocity = velocity[:, [0, 1, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_z_vel.shape)
    root_data = np.concatenate([r_velocity, l_velocity], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    # data's initial shape: (F-1, 4)
    # Contains: [root_angular_vel_y (1), root_linear_vel_xyz (3)]
    data = root_data

    # ric_data[:-1] shape: (F-1, (J-1)*3)
    # Contains: Local positions of all non-root joints relative to the root.
    # data's new shape: (F-1, 4 + (J-1)*3)
    data = np.concatenate([data, ric_data[:-1]], axis=-1)

    # rot_data[:-1] shape: (F-1, (J-1)*6)
    # Contains: Continuous 6D rotation representation for all non-root joints.
    # data's new shape: (F-1, 4 + (J-1)*3 + (J-1)*6)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)

    # local_vel shape: (F-1, J*3)
    # Contains: Velocities of all joints in the root's local coordinate system.
    # data's new shape: (F-1, 4 + (J-1)*3 + (J-1)*6 + J*3)
    data = np.concatenate([data, local_vel], axis=-1)

    # feet_l, feet_r shapes: (F-1, 2) each
    # Contains: Binary contact flags for left and right foot joints.
    # data's final shape: (F-1, 4 + (J-1)*3 + (J-1)*6 + J*3 + 4) -> (F-1, 263) for J=22
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


# In[61]:


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    """
    Reconstructs the global root orientation (as quaternion) and global root position from motion feature data.

    This function is used to recover the absolute orientation and position of the root joint for each frame,
    given a motion feature array where the root's angular velocity (around the z-axis), linear velocity (xyz-plane),
    are stored in the first four columns.

    Args:
        data (torch.Tensor): Motion feature tensor of shape (..., seq_len, feature_dim), where the first four features per frame are:
            - data[..., 0]: Root's angular velocity around the z-axis (in radians, per frame).
            - data[..., 1:4]: Root's linear velocity in the X, Y and Z directions (in the root's local frame).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - r_rot_quat: Tensor of shape (..., seq_len, 4), the global orientation of the root as a quaternion [w, x, y, z] for each frame.
            - r_pos: Tensor of shape (..., seq_len, 3), the global position of the root joint for each frame.

    Notes:
        - The function integrates the angular, linear, and z velocities over time to reconstruct the global trajectory.
        - The quaternion is constructed assuming rotation only around the z-axis (up axis).
        - All root positions (including Z) are now integrated from velocities.
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Z-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 3] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, :] = data[..., :-1, 1:4]
    '''Add Z-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    """
    Reconstructs global joint positions from a motion feature representation using 6D joint rotations.

    This function takes a motion feature tensor (as produced by process_file), the number of joints,
    and a skeleton object, and reconstructs the global 3D positions of all joints for each frame.
    It does so by:
      1. Recovering the global root orientation (as quaternion) and global root position for each frame.
      2. Extracting the 6D rotation representation for each joint from the feature vector.
      3. Concatenating the root's 6D rotation with the other joints' 6D rotations to form the full pose.
      4. Using the skeleton's forward kinematics (with 6D rotations) to compute the global joint positions.

    Args:
        data (torch.Tensor):
            A tensor of shape (..., seq_len, feature_dim) containing the motion features for each frame.
            The first four features per frame are:
                - data[..., 0]: Root's angular velocity around the z-axis (in radians, per frame).
                - data[..., 1:3]: Root's linear velocity in the X and Y directions (in the root's local frame).
                - data[..., 3]: Root's height (Z position).
            The following features include:
                - (joints_num - 1) * 3: Rotation-invariant local joint positions (not used here).
                - (joints_num - 1) * 6: 6D joint rotations for all joints except the root.
            The remaining features are local joint velocities and foot contacts (not used here).

        joints_num (int):
            The total number of joints in the skeleton (including the root).

        skeleton (Skeleton):
            An instance of a Skeleton class that provides a method
            `forward_kinematics_cont6d(cont6d_params, root_positions)` to compute global joint positions
            from 6D rotations and root positions.

    Returns:
        torch.Tensor:
            A tensor of shape (..., seq_len, joints_num, 3) containing the reconstructed global 3D positions
            of all joints for each frame.

    Notes:
        - The function assumes the feature vector layout as produced by process_file.
        - The root's 6D rotation is reconstructed from the recovered root quaternion.
        - The 6D rotations for the other joints are taken directly from the feature vector.
        - The function uses the skeleton's forward kinematics to compute the final joint positions.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_quat = torch.zeros_like(r_rot_quat).to(data.device)
    r_rot_quat[..., 0] = 1.0

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    """
    Reconstructs global joint positions from a motion feature tensor using rotation-invariant local joint positions.

    Args:
        data (torch.Tensor): Motion feature tensor of shape (..., seq_len, feature_dim), as produced by process_file.
            The first four features per frame encode root angular velocity, linear velocity (XY), and z-velocity.
            The next (joints_num - 1) * 3 features are local joint positions (excluding the root).
        joints_num (int): Total number of joints in the skeleton (including the root).

    Returns:
        torch.Tensor: Tensor of shape (..., seq_len, joints_num, 3) with reconstructed global 3D joint positions.

    Notes:
        - Only the root trajectory and local joint positions are used for reconstruction.
        - Joint rotations, velocities, and foot contacts in the feature vector are ignored.
        - Root z-position is now integrated from z-velocity instead of using absolute values.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # root_positions = torch.zeros((positions[0].shape[0], 1, 3))  # (11983, 1, 3)
    # pos_root = torch.concatenate((root_positions, positions[0]), dim=1)
    # rerun_log_joint_animation([pos_root], kinematic_chain)

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XY to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 1] += r_pos[..., 1:2]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


# In[63]:


from typing import List
import numpy as np
import rerun as rr
import time


def rerun_log_joint_animation(joint_list: List[np.ndarray],
                              kinematic_chain: List[List[int]],
                              frame_interval_sec: float = 0.00005):
    """
    Logs joint positions and bones into Rerun for interactive 3D visualization.

    yaml

    You will have fine-grained control over the time/frame displayed in the Rerun viewer.

    Args:
      joint_list (List[np.ndarray]): A list of arrays with shape (frames, num_joints, 3).
      kinematic_chain (List[List[int]]): A list of chains; each chain (list of int) refers to joint indices.
      frame_interval_sec (float): Real-time interval (in seconds) to simulate between frames.
    """
    # Determine the number of frames that all sequences share.
    min_frames = min(seq.shape[0] for seq in joint_list)
    joint_list = [seq[:min_frames] for seq in joint_list]  # trim sequences if necessary

    # Initialize Rerun. (You can change "3D Joint Animation" to an app name of your choosing.)
    rr.init("3D Joint Animation", spawn=False)
    rr.serve_web()

    # Optionally, you might want to log some scene-level transforms or coordinate systems.
    # rr.log_view_coordinates("world", up="Y")

    # Loop over frames, logging the joint data with an associated timestamp.
    # (The Rerun viewer will let you scrub any time to inspect the scene.)
    for frame in range(min_frames):
        # Use the frame index as the timestamp in seconds.
        rr.set_time_sequence("frame", frame)

        # For each sequence, log the joint points and draw the kinematic chains.
        for seq_index, seq in enumerate(joint_list):
            # Retrieve the N x 3 joint coordinates for this frame.
            points = seq[frame]  # shape: (num_joints, 3)

            # Log the joints as points.
            # The namespace (here, "/s{seq_index}/joints") determines where the geometry appears.
            # color based on sequence index for differentiation.
            color_rgb = [
                (seq_index * 50 % 255),
                (seq_index * 100 % 255),
                (seq_index * 150 % 255)
            ]
            rr.log(f"sequence_{seq_index}/joints", rr.Points3D(points, radii=0.03, colors=color_rgb))

            # For each kinematic chain, log a line connecting the joints in that chain.
            # This will draw “bones” between the joints.
            for chain_index, chain in enumerate(kinematic_chain):
                # Extract the coordinates for the joints in the current chain.
                chain_points = points[chain]
                # Log the chain as a line strip.
                rr.log(f"sequence_{seq_index}/chain_{chain_index}", rr.LineStrips3D([chain_points]))


def rerun_log_root_rotation(r_rot, positions):
    """Log root rotation as arrows in Rerun for visualization"""
    rr.init("Root Rotation Arrows", spawn=False)
    rr.serve(ws_port=0, web_port=0)

    # Convert quaternions to rotation matrices and extract forward direction
    def quat_to_forward_vector(q):
        """Extract forward vector (Y+) from quaternion"""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Convert to rotation matrix and extract second column (Y+ direction)
        forward_x = 2 * (x * y + w * z)
        forward_y = 1 - 2 * (x * x + z * z)  # Correct formula
        forward_z = 2 * (y * z - w * x)
        return np.stack([forward_x, forward_y, forward_z], axis=1)

    forward_vectors = quat_to_forward_vector(r_rot)
    root_positions = positions[:, 0]  # Root joint positions

    # Log arrows showing the facing direction
    for frame in range(len(r_rot)):
        rr.set_time_sequence("frame", frame)

        rr.log("root_rotation/arrow",
               rr.Arrows3D(
                   origins=[root_positions[frame]],
                   vectors=[forward_vectors[frame] * 0.5],
                   colors=[255, 0, 0]  # Red arrows
               ))


# In[72]:


def save_joints_wrapper(momentum_motion_path):
    try:
        save_joints(momentum_motion_path)
        return None  # Return None on success
    except Exception as e:
        # Capture the error and the traceback to be returned to the main process
        error_info = (momentum_motion_path, str(e), traceback.format_exc())
        return error_info


def save_all_joints(all_nymeria_data_paths, num_processes=4):
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create an iterable of arguments for starmap.
        # Since our worker function takes one argument, this will be a list of tuples.
        tasks = [(path,) for path in all_nymeria_data_paths]

        # Use starmap to apply the worker function to each path
        # tqdm is used to wrap the starmap iterator for a progress bar
        results = list(tqdm(pool.starmap(save_joints_wrapper, tasks), total=len(all_nymeria_data_paths),
                            desc="Processing momentum motion files"))

    # After processing, iterate through the results to find and print any errors
    print("\n--- Processing Complete ---")
    error_count = 0
    for result in results:
        if result is not None:
            error_count += 1
            path, error_msg, stack_trace = result
            print(f"\n--- Error processing {path} ---")
            print(f"Error: {error_msg}")
            print("Stack Trace:")
            print(stack_trace)

    if error_count == 0:
        print("All files processed successfully.")
    else:
        print(f"\nEncountered {error_count} errors during processing.")


'''
For HumanML3D Dataset
'''
if __name__ == "__main__":
    from nymeria.data_provider import NymeriaDataProvider

    all_nymeria_data_paths = [p for p in NYMERIA_DATA_PATH.iterdir() if p.is_dir()]

    print(f"Found {len(all_nymeria_data_paths)} momentum model files.")

    # load data order strings sperated by new line
    # data_order_path = Path('./Nymeria/data_order.txt')
    # with open(data_order_path, 'r') as f:
    #     data_order = f.read().splitlines()

    # broken_files = {'000230.npy', '000397.npy', '000361.npy', '000360.npy', '000215.npy', '000192.npy', '000205.npy', '000359.npy', '000206.npy', '000194.npy', '000210.npy', '000343.npy', '000208.npy', '000367.npy', '000222.npy', '000295.npy', '000225.npy', '000048.npy', '000201.npy', '000204.npy', '000234.npy', '000370.npy', '000193.npy', '000191.npy', '000190.npy', '000207.npy', '000369.npy', '000221.npy', '000209.npy', '000296.npy'}
    # broken_files = set([f[:-4] for f in broken_files])  # remove .npy extension
    # broken_file_names = [data_order[int(f)] for f in broken_files]

    # save_all_joints(all_nymeria_data_paths, num_processes=32)

    data_order = []
    # upper legs (b_r_leg, b_l_leg)
    l_idx1, l_idx2 = 23, 29
    # Right/Left foot (b_r_ball, b_l_ball)
    fid_r, fid_l = [27], [33]
    # Face direction, r_hip (b_r_upleg), l_hip (b_l_upleg), shoulder_r (b_r_shoulder), shoulder_l (b_l_shoulder)
    face_joint_indx = [22, 29, 8, 14]
    # r_hip, l_hip
    # r_hip, l_hip = 22, 29
    joints_num = 34
    # ds_num = 8
    data_dir = NYMERIA_DATA_PATH
    save_dir1 = Path('./Nymeria/new_joints/')
    save_dir2 = Path('./Nymeria/new_joint_vecs/')

    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    source_list = sorted(list(NYMERIA_DATA_PATH.rglob('**/joint_positions.npy')))
    print('Total files: %d' % len(source_list))

    frame_num = 0
    for i, source_file in tqdm(enumerate(source_list)):
        # if "20231006_s0_david_vega_act0_6p16ou" not in str(source_file):
        #     continue
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        if np.isnan(source_data).any():
            raise ValueError(f"NaN values found in {source_file}")
        print(f"source file: {source_file}, shape: {source_data.shape}")
        # try:
        data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
        rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
        name = str(i).zfill(6) + '.npy'
        np.save(pjoin(save_dir1, name), rec_ric_data.squeeze().numpy())
        np.save(pjoin(save_dir2, name), data)
        frame_num += data.shape[0]

        data_order.append(source_file.parent.parent.name)

        # for debugging plot original source data and the recovered data
        # plot_joint_animation([source_data, rec_ric_data.squeeze().numpy()], kinematic_chain)
        # rerun_log_joint_animation([source_data, rec_ric_data.squeeze().numpy()], kinematic_chain)
        # print(source_data)
        # plot_joint_animation([source_data], kinematic_chain)
        # except Exception as e:
        #     print(source_file)
        #     print(e)
    #         print(source_file)
    #         break

    # save the order of the files
    with open(os.path.join(save_dir1.parent, 'data_order.txt'), 'w') as f:
        for item in data_order:
            f.write("%s\n" % item)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 20 / 60))

