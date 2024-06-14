# standard libraries
import os
import logging

logger = logging.getLogger(__name__)


# Very hacky way
def is_HPC() -> bool:
  current_path = os.getcwd()
  if "evs" in current_path:
    return False
  elif "zhome" in current_path:
    return True
  else:
    raise ValueError("Unknown computing environment")

# Very hacky way
def is_VM() -> bool:
  return os.path.isdir('/home/evs/MT')

