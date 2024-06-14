# standard libraries
import logging
from typing import List
from enum import Enum

# third-party libraries
from omegaconf import DictConfig
from pydantic import BaseModel
from sglang.srt.constrained import build_regex_from_object


logger = logging.getLogger(__name__)


def local_tests(cfg: DictConfig):
  logger.notice('BEGIN task: local_tests...')

  class JSONStructure(BaseModel):
      PER: List[str] = []
      ORG: List[str] = []

  class Weapon(str, Enum):
      sword = "sword"
      axe = "axe"
      mace = "mace"
      spear = "spear"
      bow = "bow"
      crossbow = "crossbow"

  class Wizard(BaseModel):
    name: str
    age: int
    weapon: Weapon

  regex=build_regex_from_object(JSONStructure)

  print(regex)


  logger.notice('END task: local_tests.')