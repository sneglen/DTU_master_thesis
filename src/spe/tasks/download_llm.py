# standard libraries
from omegaconf import DictConfig
import logging

# third-party libraries
from transformers import AutoTokenizer, AutoModelForCausalLM

# local libraries
from src.utils.misc import is_VM

logger = logging.getLogger(__name__)


def download_llm(cfg: DictConfig):

  # Fetch model_name and model_dir from Hydra config  
  model_tag = cfg.llm.tag.for_download
  model_name = cfg.llm[model_tag].name
  model_dir = cfg.llm[model_tag].dir

  logger.notice("LLM name: " + model_name)
  logger.notice("LLM dir:  " + model_dir)

  if is_VM():
    logger.notice("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(model_dir)
    logger.notice("Model saved.")
  else:
    logger.notice("Skipping LLM download. Not intended to be run on local machine.")

  logger.notice("Downloading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.save_pretrained(model_dir)
  logger.notice("Tokenizer saved.")
