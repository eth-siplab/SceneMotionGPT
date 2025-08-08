import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio
from mGPT.data.nymeria.scripts.generate_motion_representation_nymeria import kinematic_chain as kinematic_chain_nymeria


def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    joints, out_name, title = args
    
    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    data = joints.copy().reshape(len(joints), -1, 3)

    nb_joints = joints.shape[1]
    if nb_joints == 21:
        smpl_kinetic_chain = [
            [0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4],
            [3, 5, 6, 7], [3, 8, 9, 10]
        ]
    elif nb_joints == 35:
        smpl_kinetic_chain = kinematic_chain_nymeria
    else:
        smpl_kinetic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                               [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                               [9, 13, 16, 18, 20]]
    limits = 1000 if nb_joints == 21 else 2

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    colors = [
        'red', 'blue', 'black', 'red', 'blue', 'darkblue', 'darkblue',
        'darkblue', 'darkblue', 'darkblue', 'darkred', 'darkred', 'darkred',
        'darkred', 'darkred'
    ]
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        def init():
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [[minx, miny, minz], [minx, miny, maxz],
                     [maxx, miny, maxz], [maxx, miny, minz]]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(480 / 96., 320 / 96.),
                         dpi=96) if nb_joints == 21 else plt.figure(
                             figsize=(10, 10), dpi=96)
        # fig.tight_layout()
        if title is not None:
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        init()

        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0,
                     MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0],
                      np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1],
                      linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0],
                      data[index, chain, 1],
                      data[index, chain, 2],
                      linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()

        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]),
                                       int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None):

    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size):
        out.append(
            plot_3d_motion([
                smpl_joints_batch[i], None,
                title_batch[i] if title_batch is not None else None
            ]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), duration=50)
    out = torch.stack(out, axis=0)
    return out


def draw_side_by_side(joints_list, title_list=None, outname=None, figsize=(20, 10), fps=30):
    """
    Plot multiple joint sequences side by side and save as a video/gif.

    Args:
        joints_list: List of joint tensors, each with shape (num_frames, num_joints*3)
        title_list: Optional list of titles for each sequence
        outname: Output filename for the video/gif
        figsize: Figure size
        fps: Frames per second for the output video

    Returns:
        Tensor containing the rendered frames
    """
    matplotlib.use('Agg')

    n_sequences = len(joints_list)
    if title_list is None:
        title_list = [None] * n_sequences

    # Ensure all sequences have the same number of frames
    frame_number = min([len(joints) for joints in joints_list])

    # Process joint data
    processed_data = []
    for joints in joints_list:
        data = joints.copy().reshape(len(joints), -1, 3)
        nb_joints = data.shape[1]

        # Select appropriate kinematic chain
        if nb_joints == 21:
            smpl_kinetic_chain = [
                [0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4],
                [3, 5, 6, 7], [3, 8, 9, 10]
            ]
        elif nb_joints == 34:
            smpl_kinetic_chain = kinematic_chain_nymeria
        else:
            smpl_kinetic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                                  [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                                  [9, 13, 16, 18, 20]]

        # Normalize the data
        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]

        processed_data.append({
            'data': data,
            'trajec': trajec,
            'MINS': MINS,
            'MAXS': MAXS,
            'kinetic_chain': smpl_kinetic_chain,
            'nb_joints': nb_joints
        })

    colors = [
        'red', 'blue', 'black', 'red', 'blue', 'darkblue', 'darkblue',
        'darkblue', 'darkblue', 'darkblue', 'darkred', 'darkred', 'darkred',
        'darkred', 'darkred'
    ]

    radius = 4

    def update(index):
        fig = plt.figure(figsize=figsize, dpi=96)

        for seq_idx in range(n_sequences):
            ax = fig.add_subplot(1, n_sequences, seq_idx + 1, projection='3d')

            # Get data for current sequence
            seq_data = processed_data[seq_idx]
            data = seq_data['data']
            trajec = seq_data['trajec']
            MINS = seq_data['MINS']
            MAXS = seq_data['MAXS']
            smpl_kinetic_chain = seq_data['kinetic_chain']
            nb_joints = seq_data['nb_joints']

            # Set up axis
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])
            ax.grid(b=False)

            # Plot title if provided
            if title_list[seq_idx] is not None:
                title_sp = title_list[seq_idx].split(' ')
                if len(title_sp) > 20:
                    title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
                elif len(title_sp) > 10:
                    title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
                else:
                    title = title_list[seq_idx]
                ax.set_title('\n'.join(wrap(title, 40)), fontsize=12)

            # Plot XZ plane
            verts = [[-radius / 2, 0, -radius / 2], [-radius / 2, 0, radius / 2],
                     [radius / 2, 0, radius / 2], [radius / 2, 0, -radius / 2]]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

            # View settings
            ax.view_init(elev=110, azim=-90)
            ax.dist = 7.5

            # Plot trajectory
            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0],
                          np.zeros_like(trajec[:index, 0]),
                          trajec[:index, 1] - trajec[index, 1],
                          linewidth=1.0,
                          color='blue')

            # Plot skeleton
            for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
                linewidth = 4.0 if i < 5 else 2.0
                ax.plot3D(data[index, chain, 0],
                          data[index, chain, 1],
                          data[index, chain, 2],
                          linewidth=linewidth,
                          color=color)

            # Remove axis labels and ticks
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        # plt.tight_layout()

        # Save to buffer
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=96)
        io_buf.seek(0)
        arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]),
                                   int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        plt.close()
        return arr

    # Render all frames
    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)

    # Save to video/gif if outname is provided
    if outname is not None:
        writer = imageio.get_writer(outname, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
        for frame in out:
            writer.append_data(frame)
        writer.close()

    return torch.from_numpy(out)