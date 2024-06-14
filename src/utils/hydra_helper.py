# standard libraries
import os
import logging


# third-party libraries
import hydra
from functools import wraps
from omegaconf import DictConfig
from hydra import initialize, compose

# local libraries
from src.config import logging_config


def get_data_dir(cfg: DictConfig):
  logger = logging.getLogger(logging_config.logger_name)
  # Check if overrule_data_dir is set otherwise use hydras own output_dir
  if cfg.data.overrule_data_dir != []:
    data_dir = cfg.data.overrule_data_dir
    if cfg.main_task.openai_annotate=='run' and cfg.openai.query_online:
      logger.notice("Running OFFLINE now (contradictory settings): <cfg.openai.query_online> enabled but <overrule_data_dir> is not empty.")
      cfg.openai.query_online = False
  else:
    data_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

  return data_dir


def parse_article_DR_indices(range_list):
    """Parse the list of article indices and return a list of integers.
       Example: 
              yaml entry:
                art_to_annotate: ["0-1", "5", "7-10"]
              
              art_to_annotate = cfg['data']['art_to_annotate']
              indices = hh.parse_article_indices(art_to_annotate)
              [0, 1, 5, 7, 8, 9, 10]
    """
    indices = []
    for item in range_list:
        if '-' in item:  # Check if the item is a range
            start, end = item.split('-')
            indices.extend(range(int(start), int(end) + 1))  # +1 because range is exclusive at the end
        else:
            indices.append(int(item))
    return indices


def can_run_as_standalone(config_path):
  def decorator(func):
      """Decorator to run a function as standalone.
      The function will be run as standalone if the cfg and hydra_output_dir are not provided.
      hydra_output_dir is fetched from conf/config.yaml file.
      """
      @wraps(func)
      def wrapper(cfg: DictConfig = None, hydra_output_dir: str = None, *args, **kwargs):
          try:
            if cfg is None or hydra_output_dir is None:
                with initialize(version_base=None, config_path=config_path):
                    cfg = compose(config_name="config.yaml")

                hydra_output_dir = cfg['standalone_dir']
                if not os.path.exists(hydra_output_dir):
                    os.makedirs(hydra_output_dir)
          except Exception as e:
            raise Exception(f"Hydra initialization: {e}\nVerify that <config_path> is at the correct directory hierarchy.")

          return func(cfg, hydra_output_dir, *args, **kwargs)
      return wrapper
  return decorator
