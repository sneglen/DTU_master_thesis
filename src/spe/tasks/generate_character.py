import sglang as sgl
from omegaconf import DictConfig
import logging


logger = logging.getLogger(__name__)

character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)

@sgl.function
def worker_generate_character(s, name:str):

  s += name + " is a character in Harry Potter. Please fill in the following information about this character in JSON format.\n"
  logger.notice(f'LLM request: {s}:\nregex: {character_regex}\n')
  s += sgl.gen("json_output", max_tokens=256, regex=character_regex)
       

def generate_character(cfg: DictConfig):
  try:
    logger.notice("BEGIN task: generate_character...")
    state = worker_generate_character.run(name="Lord VoldeMort")

    logger.notice("Generated character:")
    print(state.text())    

  except Exception as e:
    logger.error(f"Error: {e}") 

  finally:
    logger.notice("END task: generate_character.")