# third-party libraries
import pytest as pt
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

import pytest
from src.config.logging_config import initiate_logging
#from tests.load_hydra_for_testing import hydra_cfg as load_hydra_cfg


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


@pytest.fixture(scope="session")
def logger(hydra_cfg):
    return initiate_logging(hydra_cfg)
