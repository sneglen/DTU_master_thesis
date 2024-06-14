## pytest -s tests/finetune/test_load_data_from_keys_in_chatML_format.py

# standard libraries

# third-party libraries

# local libraries
from tests.load_hydra_for_testing import load_hydra_config
from src.finetune import finetune_data_utils as ft_du

## Execution hint (from project root):
#  pytest -s tests/finetune

def test_load_data_from_keys_in_chatML_format(logger):

  cfg = load_hydra_config()
  cfg.llm.tag.for_finetuning = 'munin'
  cfg.finetuning.training_settings.indices.train = ['0','1','2']
  cfg.finetuning.training_settings.pct_of_datasize = 100
  cfg.finetuning.training_settings.shuffle = False  

  training_keys = ['PER', 'ORG', 'QUOTES', 'SRC', 'GEN', 'FUN', 'ROL', 'EMP']
  dataset = ft_du.load_data_from_keys_in_chatML_format(cfg, keys=training_keys, selected_split='train')

## PER check
  EXP_PER =  "Alison Van Uytvanck\nCaroline Wozniacki\nJohanna Larsson"
  actual_PER = dataset[0]['messages'][2]['content']

  assert actual_PER == EXP_PER, f"Actual PER: <{actual_PER}> does not match expected PER: <{EXP_PER}>"

## ORG check
  EXP_ORG =  "Australian Open"
  actual_ORG = dataset[1]['messages'][2]['content']

  assert actual_ORG == EXP_ORG, f"Actual ORG: <{actual_ORG}> does not match expected ORG: <{EXP_ORG}>"

## QUOTE check
  EXP_quote = 'Forhåbentlig bliver '
  actual_quote = dataset[3]['messages'][2]['content'][:20]

  assert actual_quote == EXP_quote, f"Actual quote: {actual_quote} does not match expected quote"

## SRC check
  EXP_SRC = 'Caroline Wozniacki'
  actual_SRC = dataset[7]['messages'][2]['content']

  assert actual_SRC == EXP_SRC, f"Actual SRC: <{actual_SRC}> does not match expected SRC: <{EXP_SRC}>"

## GEN check
  EXP_GEN = 'F'
  actual_GEN = dataset[12]['messages'][2]['content']

  assert actual_GEN == EXP_GEN, f"Actual GEN: <{actual_GEN}> does not match expected GEN: <{EXP_GEN}>"

## FUN check
  EXP_FUN = 'Other'
  actual_FUN = dataset[17]['messages'][2]['content']
  
  assert actual_FUN == EXP_FUN, f"Actual FUN: <{actual_FUN}> does not match expected FUN: <{EXP_FUN}>"

## ROL check
  EXP_ROL = 'tennisstjerne'
  actual_ROL = dataset[22]['messages'][2]['content']
  
  assert actual_ROL == EXP_ROL, f"Actual ROL: <{actual_ROL}> does not match expected ROL: <{EXP_ROL}>"

## EMP check
  EXP_EMP = ''
  actual_EMP = dataset[27]['messages'][2]['content']
  
  assert actual_EMP == EXP_EMP, f"Actual EMP: <{actual_EMP}> does not match expected EMP: <{EXP_EMP}>"

