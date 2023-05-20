import numpy as np
import os


def readSkeleton(filename):
    # Read skeleton positions from a text file.
    # filename: text file of the skeleton positions.
    # skeleton: skeleton positions.
    if filename is None:
        message = 'Missing file name!'
        print(message)
        return -1, -1

    try:
        with open(filename, 'r') as fid:
            tmp = np.fromfile(fid, dtype=int, count=2, sep=' ')
            num_frames, num_joints = tmp[0], tmp[1]
            skeleton = np.zeros((num_frames, num_joints * 2, 4))
            for f in range(num_frames):
                num_joints_current_frame = np.fromfile(fid, dtype=int, count=1, sep=' ')
                skeleton_current_frame = np.fromfile(fid, dtype=float, count=num_joints_current_frame[0] * 4, sep=' ')
                skeleton_current_frame = skeleton_current_frame.reshape(num_joints_current_frame[0], 4)
                if num_joints_current_frame[0] >= num_joints * 2:
                    skeleton[f, :, :] = skeleton_current_frame[:num_joints * 2, :]
    except FileNotFoundError:
        message = 'Cannot open the file: ' + filename
        print(message)
        return -1, -1

    return skeleton, num_frames


def saveSkeletonMatrix(conf, parent_dir, save_dir):
    skeleton_all = np.empty((conf.num_actions, conf.num_subjects, conf.num_env), dtype=np.object)

    # Loop over actions, subjects, and environments
    for a in range(1, conf.num_actions + 1):
        for s in range(1, conf.num_subjects + 1):
            for e in range(1, conf.num_env + 1):
                # Load the skeleton data from the text file
                filename = os.path.join(parent_dir, f'a{a:02d}_s{s:02d}_e{e:02d}_skeleton.txt')
                skeleton = readSkeleton(filename)

                # Store the skeleton data in the cell array
                skeleton_all[a - 1, s - 1, e - 1] = skeleton

    # Save the cell array to a .mat file
    np.savez(f'{save_dir}/skeletons_matrix.npz', skeleton_all=skeleton_all)


def computePairwiseJointPositions(skeleton, joint_id, target_ids):
    # Compute the pairwise joint positions from joint joint_id to all the joints in target_ids.
    # skeleton: [#joints, D] ~ [20, 4]
    # joint_id: int     e.g. [HEAD]
    # target_ids: list      conf.target_list
    jnts = skeleton[:, 0:3]
    normalization_factor = 1
    count = 0
    angles = np.zeros(sum(np.array(target_ids) != joint_id) * 3)

    for t in range(len(target_ids)):
        target_id = target_ids[t]
        if (target_id == joint_id):
            continue
        target = jnts[joint_id-1, :] - jnts[target_id-1, :]     # [3]
        angles[count:count + 3] = target / normalization_factor
        count = count + 3

    return angles   # [39]


def fftPyramid(data_in, num_samples, use_cos_sin=False):
    # Compute the Fourier temporal pyramid features.
    # data_in: input features, where the first dimension is the number of
    # the data, and the second dimension is the feature dimension.
    # num_samples: number of low frequency coefficients for features.
    # FFT is applied to each dimension of data_in.
    if not use_cos_sin:
        use_cos_sin = False

    # compute cosin and sin for the features, only good for angles.
    if use_cos_sin == 1:
        data_cos = np.cos(data_in)
        data_cos[1::2, :] = data_cos[1::2, :] / 2
        data_sin = np.sin(data_in[1::2, :]) / 2
        data_processed = np.concatenate((data_cos, data_sin), axis=0)
    else:
        data_processed = data_in

    # padding the data if needed.
    padding_num = num_samples - data_processed.shape[1]
    if padding_num > 0:
        data_processed = np.concatenate((data_processed,
                                          np.zeros((data_processed.shape[0], padding_num))),
                                          axis=1)

    num_segments = 3
    data_length = data_processed.shape[1]
    segments_data = np.arange(1, data_length // num_segments + 1) * num_segments
    segments_data = np.concatenate((segments_data, [data_length + 1]))
    segments_samples = num_samples // num_segments

    if len(segments_data) > num_segments + 1:
        print('error: input feature length is too short.')

    segments_all = []

    # compute the pyramid.
    for i in range(num_segments):
        data_segments = data_processed[:, segments_data[i] - num_segments: segments_data[i+1] - num_segments]
        data_segments_fft = np.zeros_like(data_segments)

        for j in range(data_segments.shape[0] // 3):
            range_ = slice(j * 3, (j + 1) * 3)
            data_segments_fft[range_, :] = np.fft.fft(data_segments[range_, :].T).T
            data_segments_fft[:, :] = np.abs(data_segments_fft[:, :])

        segments_all.append(data_segments_fft[:, :segments_samples//2])
        segments_all.append(data_segments_fft[:, -segments_samples//2:])

    data_fft = np.fft.fft(data_processed.T).T
    data_fft[:, :] = np.abs(data_fft[:, :])

    features_all = np.concatenate(segments_all + [data_fft[:, :num_samples//2], data_fft[:, -num_samples//2:]], axis=1)
    feature = features_all * 100 / data_in.shape[1]

    return feature
