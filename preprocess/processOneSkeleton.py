import numpy as np
import os
from config import configDailyAcitity

from data_utils import computePairwiseJointPositions
from data_utils import fftPyramid


def processOneSkeleton(a, s, e, recompute_features, conf, data_dir):
    # Compute the skeleton feature for one depth sequence.
    # a, s, e: the action, subject, and enviornment id.
    # recompute_features: if set to 1, the feature is computed
    # regardless of whether the feature file already exists.

    # Hack for parsing parameters from command line if the function
    # is compiled with Matlab compiler.
    if isinstance(a, str):
        a = int(a)
    if isinstance(s, str):
        s = int(s)
    if isinstance(e, str):
        e = int(e)
    if isinstance(recompute_features, str):
        recompute_features = int(recompute_features)

    conf = conf
    data = np.load(data_dir)
    skeleton_all = data

    # The weights for the skeleton.
    target_list = conf.target_list
    weight_list = conf.weight_list
    skeleton = skeleton_all[a, s, e]    # [143, 40, 4]
    num_frame = skeleton.shape[0]
    num_joint = skeleton.shape[1]

    pivot_shape = (skeleton.shape[2]-1)*sum(np.array(target_list)!=conf.joints_all[0])  # 3 * 13
    angles = np.empty((num_frame, pivot_shape * 2, len(conf.joints_all)))   # [143, 78, 10]
    num_samples = 10
    avg_positions = np.empty((num_frame, 2))    # [143, 2]

    for f in range(num_frame):
        # Normalize the absolute positions.
        skeleton[f, 1::2, 2] = skeleton[f, 1::2, 2] * 0.01
        positions = np.empty((pivot_shape, len(conf.joints_all)))   # [39, 10]
        # The odd dimension of the feature is relative skeleton positions.
        skeleton_relative = np.reshape(skeleton[f, 0::2, :], (num_joint // 2, skeleton.shape[2]))   # [20, 4]
        for j in range(len(conf.joints_all)):
            position = computePairwiseJointPositions(skeleton_relative, conf.joints_all[j], target_list) * weight_list[j]
            positions[:, j] = position  # 3*k

        positions2 = np.copy(positions)
        # The even dimension of the feature is absolute skeleton positions.
        skeleton_absolute = np.reshape(skeleton[f, 1::2, :], (num_joint // 2, skeleton.shape[2]))
        for j in range(len(conf.joints_all)):
            position = computePairwiseJointPositions(skeleton_absolute, conf.joints_all[j], target_list) * weight_list[j]
            positions2[:, j] = position

        # positions: [pivot_shape, joints_all]
        avg_position = np.mean(skeleton[f, 1::2, 0:3], axis=0)
        avg_position = avg_position.reshape((1, 3)) * 10
        avg_position = avg_position[0, 0:2]
        angles[f, :, :] = np.concatenate((positions, positions2), axis=0)
        avg_positions[f, :] = avg_position

    # angles: [#frames, 2 * pivot_shape, joints_all]    ~ [143, 78, 10]
    feature_current = np.empty()
    for j in range(len(conf.joints_all)):
        angles_jt = angles[:, :, j]
        angles_jt = np.reshape(angles_jt, (angles_jt.shape[0], angles_jt.shape[1]))
        angles_jt = angles_jt.T     # [2 * pivot_shape, #frames]  ~ [78, 143]
        angle_cos_sin = fftPyramid(angles_jt, num_samples).real      # 2P, S
        angles_shift = np.roll(angles_jt, 1, axis=1)    # 2P, F
        angles_diff = angles_jt - angles_shift
        angles_diff = angles_diff[:, :-1]
        feature_diff = fftPyramid(angles_diff, num_samples).real     # 2P, S
        if j == 0:
            feature_current = np.empty((feature_diff.shape[0], 2 * feature_diff.shape[1], len(conf.joints_all)))
        feature_current[:, :, j] = np.concatenate((angle_cos_sin, feature_diff), axis=1)    # 2P, 2S

    angles_jt = avg_positions.T     # 2, F
    angle_cos_sin = fftPyramid(angles_jt, num_samples).real     # 2, 22
    angles_shift = np.roll(angles_jt, shift=1, axis=1)      # 2, 143
    angles_diff = angles_jt - angles_shift      # 2, 143
    angles_diff = angles_diff[:, :-1]   # 2, 142
    feature_diff = fftPyramid(angles_diff, num_samples).real     # 2, 22

    pos = np.concatenate((angle_cos_sin, feature_diff), axis=1)     # 2, 2*S

    filename_out = os.path.join(conf.feature_dir, f'a{a:02d}_s{s:02d}_e{e:02d}_skeleton.mat')
    # scipy.io.savemat(filename_out, {'feature_current': feature_current, 'pos': pos})
    np.savez(f'{filename_out}', feature_current=feature_current, pos=pos)

