# standard libraries
from enum import Enum

# third-party libraries
from pydantic import BaseModel, conlist
from sglang.srt.constrained import build_regex_from_object

# local libraries

## PER
class per_object(BaseModel):
  PER: conlist(str, min_length=0, max_length=15) = []

per_regex = build_regex_from_object(per_object)

## ORG
class org_object(BaseModel):
  ORG: conlist(str, min_length=0, max_length=15) = []

org_regex = build_regex_from_object(org_object)

## QUOTES
class quote_text_object(BaseModel):
    TEXT: str

class quotes_text_object(BaseModel):
  QUOTES: conlist(quote_text_object, min_length=0, max_length=15) = []

quotes_text_regex = build_regex_from_object(quotes_text_object)

## SRC
class src_object(BaseModel):
  SRC: conlist(str, min_length=0, max_length=1) = []

src_regex = build_regex_from_object(src_object)

## GEN
class gen_choices_object(str, Enum):
    Male = "M"
    Female = "F"
    Unknown = "X"

class gen_object(BaseModel):
  GEN: gen_choices_object = gen_choices_object.Unknown

gen_regex = build_regex_from_object(gen_object)

## FUN
class fun_choices_object(str, Enum):
    Expert = "Expert"
    Case = "Case"
    Politician = "Politician"
    DRSource = "DR source"
    InterestOrganization = "Interest organization"
    ProfessionalExpert = "Professional expert"
    Authority = "Authority"
    Other = "Other"

class fun_object(BaseModel):
  FUN: fun_choices_object = fun_choices_object.Other

fun_regex = build_regex_from_object(fun_object)


## ROL
class rol_object(BaseModel):
  ROL: conlist(str, min_length=0, max_length=2) = []

rol_regex = build_regex_from_object(rol_object)


## EMP
class emp_object(BaseModel):
  EMP: conlist(str, min_length=0, max_length=1) = []

emp_regex = build_regex_from_object(emp_object)


## QUOTE (unit)
class quote_unit_object(BaseModel):
  text: str = ""
  SRC: conlist(str, min_length=0, max_length=1) = []
  GEN: gen_choices_object = gen_choices_object.Unknown
  FUN: fun_choices_object = fun_choices_object.Other
  ROL: conlist(str, min_length=0, max_length=2) = []
  EMP: conlist(str, min_length=0, max_length=1) = []


def get_default_dict_for_key(key: str):
  defaults = {
    "PER": [],
    "ORG": [],
    "QUOTES": [],
    "SRC": [],
    "GEN": gen_choices_object.Unknown.value,
    "FUN": fun_choices_object.Other.value,
    "ROL": [],
    "EMP": []
  }
  return defaults.get(key, None)
