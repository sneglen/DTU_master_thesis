# standard libraries
import logging
import socket
import time


# third-party libraries
from omegaconf import DictConfig
from sglang import set_default_backend, RuntimeEndpoint


logger = logging.getLogger(__name__)


def load_spe_instructions(cfg: DictConfig) -> str:
    """Load system content from file."""

    instructions_file = f"{cfg.spe.query_dir}{cfg.spe.query_name}_v{cfg.spe.query_ver}.txt"

    try:
        with open(instructions_file, 'r') as file:
            spe_instructions = file.read()
    except FileNotFoundError:
          msg = f"File not found: {instructions_file}"
          logger.error(msg)
          raise FileNotFoundError(msg)
    except Exception as e:
          msg = f"An error occurred while reading the file: {e}"
          logger.error(msg)
          raise Exception(msg)
                            
    return spe_instructions


def wait_for_llm_launch(port=30000, timeout=3000, interval=5):
    logger.notice("Waiting for LLM server to be launched...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if result == 0:
            # Port is ACTIVE
            logger.notice(f"LLM launched - elapsed time: {time.time() - start_time:.0f}[s]")
            return True
        time.sleep(interval)
    # Timeout: Port is NOT ACTIVE    
    return False



def set_llm_backend() -> RuntimeEndpoint:

  try:
    RuntimeEndpoint_obj = RuntimeEndpoint("http://localhost:30000")
    set_default_backend(RuntimeEndpoint_obj)

    logger.notice(f"LLM: {RuntimeEndpoint_obj.get_model_name()} running...")
    return RuntimeEndpoint_obj
  
  except Exception as e:
    logger.error(f"Error: {e}")
    return None
