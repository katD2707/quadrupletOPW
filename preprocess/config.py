from dataclasses import dataclass


@dataclass
class configDailyAcitity:
    num_actions = 1, 6
    num_subjects = 1, 0
    num_env = 2

    data_dir = ''
    feature_dir = ''

    SHOULDER_LEFT = 5
    SHOULDER_RIGHT = 9
    HIP = 1
    SPINE = 2
    NECK = 3
    HEAD = 4
    ELBOW_LEFT = 6
    ELBOW_RIGHT = 10
    WRIST_LEFT = 7
    WRIST_RIGHT = 11
    HIP_LEFT = 13
    HIP_RIGHT = 16
    KNEE_LEFT = 15
    ANKLE_LEFT = 14
    KNEE_RIGHT = 17
    ANKLE_RIGHT = 18

    joints_all = [
        HEAD, NECK, ANKLE_LEFT, ANKLE_RIGHT, ELBOW_LEFT, WRIST_LEFT,
        ELBOW_RIGHT, WRIST_RIGHT, SHOULDER_LEFT, SHOULDER_RIGHT,
    ]
    target_list = [
        HEAD, NECK, ANKLE_LEFT, ANKLE_RIGHT, ELBOW_LEFT, WRIST_LEFT,
        ELBOW_RIGHT, WRIST_RIGHT, SHOULDER_LEFT, SHOULDER_RIGHT,
        HIP, KNEE_LEFT, KNEE_RIGHT, SPINE,
    ]
    weight_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    recompute_features = 1
    num_bins = [14, 14, 1]
    bin_size = [6, 6, 80]
    saturation_size = 3
    num_samples_SOP = 10

    train_sub = [1, 3, 5, 7, 9]
    test_sub = [2, 4, 6, 8, 10]
