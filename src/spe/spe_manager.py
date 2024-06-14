# standard libraries
import logging

# third-party libraries
from omegaconf import DictConfig

# local libraries
from src.spe.tasks import task
import src.spe.spe_utils as mu
from src.utils.misc import is_VM


logger = logging.getLogger(__name__)


def run_tasks(cfg: DictConfig):

## Tasks that do not depend on a running LLM... 
  if cfg.spe.task.download_llm == 'run' :
    task.download_llm(cfg=cfg)

  elif cfg.spe.task.local_tests == 'run':
    task.local_tests(cfg)

  elif not is_VM() and cfg.spe.task.annotate_articles == 'run':
    # Allow to run in  local PC (waiting for LLM to be launched is therefore skipped) 
    logger.notice(f"\n*** OFFLINE MODE (VM: {is_VM()}) - annotate_articles() ***\n")
    task.annotate_articles(cfg=cfg)

## Following tasks require a running LLM...
  else:  
    # Launch LLM and set backend
    llm_launched = mu.wait_for_llm_launch()
    RuntimeEndpoint_obj = mu.set_llm_backend()

    # Run tasks
    if llm_launched and RuntimeEndpoint_obj is not None:
      
      if cfg.spe.task.hello_llm == 'run':
        task.hello_llm(cfg)

      if cfg.spe.task.multiturn_questions == 'run':
        task.multiturn_questions(cfg)

      if cfg.spe.task.generate_character == 'run':
        task.generate_character(cfg)

      if cfg.spe.task.annotate_articles == 'run':
        task.annotate_articles(cfg=cfg)

if __name__ == "__main__":
  pass
  # TODO: Fetch hydra's cfg object and pass it to run_tasks()
  #run_tasks(cfg)
