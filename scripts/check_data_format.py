from pathlib import Path
from typing import List
import numpy as np
import rerun as rr
import rerun.datatypes as rrd
import time

import torch

from mGPT.data.nymeria.scripts.generate_motion_representation_nymeria import recover_from_ric, kinematic_chain



def rerun_log_joint_animation(joint_list: List[np.ndarray],
    kinematic_chain: List[List[int]]):
    """
    Logs joint positions and bones into Rerun and adds a camera that follows the first skeleton.

    How to use the camera:
    1. Run this script to launch the Rerun viewer.
    2. In the top-right of the 3D Viewport panel, click the "eye" icon (View Layout).
    3. Set the "Origin" of the view to "/camera".
    The viewport will now render from the perspective of the moving camera.

    Args:
      joint_list (List[np.ndarray]): A list of arrays with shape (frames, num_joints, 3).
      kinematic_chain (List[List[int]]): A list of chains; each chain (list of int) refers to joint indices.
    """
    # Determine the number of frames that all sequences share.
    min_frames = min(seq.shape[0] for seq in joint_list)
    joint_list = [seq[:min_frames] for seq in joint_list]  # trim sequences if necessary

    # Initialize Rerun.
    rr.init("3D Joint Animation with Camera", spawn=False)
    rr.serve_web(web_port=0, grpc_port=0)

    # Set the up-axis for a consistent 3D view.
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Loop over frames, logging the joint data with an associated timestamp.
    for frame in range(min_frames):
        # Use the frame index as the timestamp.
        rr.set_time("frame", sequence=frame)

        # --- Automatic Camera Control ---
        # The camera will follow the first skeleton in the list (sequence_0).
        if joint_list:
            # 1. Get the current positions of the joints for the skeleton to be tracked.
            target_points = joint_list[0][frame]

            # 2. Calculate the center of the skeleton to act as the look-at target.
            skeleton_center = np.mean(target_points, axis=0)

            # 3. Define the camera's position at a fixed offset from the skeleton's center.
            #    Adjust the x, y, z values in np.array to change the camera angle.
            camera_position = skeleton_center + np.array([0, 60.0, 100.0])

            # 4. Log the transform for the camera entity. This moves the camera.
            #    When you set the view to this entity, it will "look at" its own origin.
            rr.log("camera", rr.Transform3D(translation=camera_position))

        # --- Log Skeleton Data (your original code) ---
        # For each sequence, log the joint points and draw the kinematic chains.
        for seq_index, seq in enumerate(joint_list):
            # Retrieve the N x 3 joint coordinates for this frame.
            points = seq[frame]  # shape: (num_joints, 3)

            # Log the joints as points.
            entity_path = f"sequence_{seq_index}"
            color_rgb = [
                (seq_index * 60 % 255),
                (seq_index * 120 % 255),
                (255 - seq_index * 60 % 255)
            ]
            rr.log(f"{entity_path}/joints", rr.Points3D(points, radii=0.03, colors=color_rgb))

            # For each kinematic chain, log a line connecting the joints.
            for chain_index, chain in enumerate(kinematic_chain):
                chain_points = points[chain]
                rr.log(f"{entity_path}/chain_{chain_index}", rr.LineStrips3D([chain_points]))




if __name__ == "__main__":
    # Example usage:
    # joint_list: List of numpy arrays, each with shape (frames, num_joints, 3)

    new_joints = Path('/home/dominik/Documents/repos/MotionGPT/datasets/Nymeria/new_joints')
    new_joints_vecs = Path('/home/dominik/Documents/repos/MotionGPT/datasets/Nymeria/new_joint_vecs')
    data_order = Path("/home/dominik/Documents/repos/MotionGPT/datasets/Nymeria/data_order.txt")

    nymeria_path = Path("/media/dominik/DominikSSD/AriaData/data/all_data/")


    new_joint_files = sorted(list(new_joints.glob('*.npy')))
    new_joints_vecs_files = sorted(list(new_joints_vecs.glob('*.npy')))

    # weird rotation:  241,
    # upside down: 292, 131, 277

    # new_joint_files = new_joint_files[292:293]
    # new_joints_vecs_files = new_joints_vecs_files[292:293]

    # premute both lists in the same way
    assert len(new_joint_files) == len(new_joints_vecs_files), "Mismatch in length of joint files and joint vectors files."

    indices = np.arange(len(new_joint_files))
    np.random.shuffle(indices)
    new_joint_files = [new_joint_files[i] for i in indices]
    new_joints_vecs_files = [new_joints_vecs_files[i] for i in indices]

    data_order_list = [line.strip() for line in data_order.read_text().splitlines()]

    for i, (joint_file, joint_vecs_file) in enumerate(zip(new_joint_files, new_joints_vecs_files)):
        data_idx = int(joint_file.stem)
        data_path = Path(data_order_list[data_idx])

        print(f"Processing {i+1}/{len(new_joint_files)}: {joint_file.name}")
        # joint_data_orig = np.load(nymeria_path / data_path / "body" / "joint_positions.npy")
        joint_data = np.load(joint_file)
        joint_vecs_data = torch.from_numpy(np.load(joint_vecs_file))

        recovered_joints = recover_from_ric(joint_vecs_data, 34).numpy()

        recovered_joints = recovered_joints[5000:]

        rerun_log_joint_animation([recovered_joints], kinematic_chain)

        input("plot next? Press Enter to continue...")


