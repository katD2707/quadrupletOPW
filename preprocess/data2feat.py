from processOneSkeleton import processOneSkeleton
from config import configDailyAcitity


def main(config):
    # Compute skeleton features
    recompute_features = 1
    for a in range(config.num_actions):
        for s in range(config.num_subjects):
            for e in range(config.num_env):
                processOneSkeleton(a, s, e, recompute_features=recompute_features, conf=config)


if __name__ == "__main__":
    conf = configDailyAcitity()
    main(conf)