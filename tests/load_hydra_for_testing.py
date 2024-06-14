# third-party libraries
import pytest as pt
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize


def load_hydra_config(cfg_name="config.yaml", cfg_path="../conf"):
    """
    Manually initializes Hydra without using the decorator and fetches the configuration.
    """
    if not GlobalHydra.instance().is_initialized():
        initialize(config_path=cfg_path, job_name="testing_job", version_base=None)
    cfg = compose(config_name=cfg_name)
    return cfg

@pt.fixture(scope="session")
def hydra_cfg():
    return load_hydra_config()
