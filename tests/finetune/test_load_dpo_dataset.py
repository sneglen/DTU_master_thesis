## pytest -s tests/finetune/test_load_dpo_dataset.py

# standard libraries

# third-party libraries

# local libraries
from tests.load_hydra_for_testing import load_hydra_config
from src.finetune import finetune_data_utils as ft_du

def test_load_dpo_dataset(logger):

  cfg = load_hydra_config()
  cfg.data.overrule_data_dir = 'tmp/pytests/'
  cfg.llm.tag.for_finetuning = 'munin'   
  cfg.finetuning.training_settings.indices.train    = ['0','1','2', '88'] # 88 to test wrong format: empty PER+ORG
  cfg.finetuning.training_settings.pct_of_datasize  = 100
  cfg.finetuning.training_settings.shuffle          = False  
  cfg.finetuning.training_settings.dpo_keys         = ['PER', 'ORG', 'QUOTES', 'SRC', 'GEN', 'FUN', 'EMP', 'ROL']
  cfg.data.dpo.QUOTES.max_retrieval                 = 10 # random function generates 2 quotes with 10 as max_retrieval

  dataset = ft_du.load_dpo_dataset(cfg)


## PER check: chosen ####################################
  idx = 0
  EXP_PER_c =  '{"PER": ["Alison Van Uytvanck", "Caroline Wozniacki", "Johanna Larsson"]}'
  actual_PER_c = dataset[idx]['chosen']

  assert actual_PER_c == EXP_PER_c, f"PER check - chosen: Actual PER: <{actual_PER_c}> does not match expected PER: <{EXP_PER_c}>"

## PER check: rejected
  EXP_PER_r =  '{"PER": ["Karsten Hønge", "Ellen Trolle", "Johnny Andresen", "Thomas Mogensen", "Christina Pojezny"]}'
  actual_PER_r = dataset[idx]['rejected']

  assert actual_PER_r == EXP_PER_r, f"PER check - rejected: Actual PER: <{actual_PER_r}> does not match expected PER: <{EXP_PER_r}>"

## ORG check: chosen ####################################
  idx = 1
  EXP_ORG_c =  '{"ORG": ["Australian Open"]}'
  actual_ORG_c = dataset[idx]['chosen']

  assert actual_ORG_c == EXP_ORG_c, f"ORG check - chosen: Actual ORG: <{actual_ORG_c}> does not match expected ORG: <{EXP_ORG_c}>"

## ORG check: rejected
  EXP_ORG_r =  '{"ORG": ["Northumbria University"]}' 
  actual_ORG_r = dataset[idx]['rejected']

  assert actual_ORG_r == EXP_ORG_r, f"ORG check - rejected: Actual ORG: <{actual_ORG_r}> does not match expected ORG: <{EXP_ORG_r}>"


## QUOTES check: chosen (article contains quotes) ##########
  idx = 2
  EXP_QUOTES_c =  '{"QUOTES": [{"TEXT": "Jeg følte, at hun holdt et højt niveau. Jeg var overrasket over, at der virkelig var lange dueller, og boldene kom ikke langsomt over nettet"}, {"TEXT": "Forhåbentlig bliver det en rigtig god kamp. Vi har ikke spillet mod hinanden i årevis, men jeg kender hende rigtig godt"}, {"TEXT": "Det har ikke ændret noget i mit spil. Jeg har ikke gjort noget anderledes, føler jeg. Andet end at jeg tror, at jeg føler mig mere stolt"}, {"TEXT": "Jeg servede godt, når jeg skulle. Jeg fik brudt hende, og så var det vigtigt at holde min serv"}, {"TEXT": "Hun blev farlig til sidst, når hun begyndte at ramme linjerne og ramme hjørnerne. Så var det ikke nemt at returnere"}]}'
  actual_QUOTES_c = dataset[idx]['chosen']

  assert actual_QUOTES_c == EXP_QUOTES_c, f"QUOTES filled check - chosen: Actual QUOTES: <{actual_QUOTES_c}> does not match expected QUOTES: <{EXP_QUOTES_c}>"

## QUOTES check: rejected
  EXP_QUOTES_r =  '{"QUOTES": [{"TEXT": "Jeg følte, at hun holdt et højt niveau. Jeg var overrasket over, at d"}, {"TEXT": "Forhåbentlig bliver det en rigtig god kamp. Vi har ikke spillet mod hinanden i årevis, men jeg kender hende rigtig godt, siger Caroline Wo"}, {"TEXT": "holde på afstand.\n- Det har ikke ændret noget i mit spil. Jeg har ikke gjort noget anderledes, føler jeg. Andet end at jeg tror, at jeg føler mig mere stol"}, {"TEXT": "Jeg servede godt, når jeg skulle. Jeg fik brudt hende, og så var det vigtigt at holde min serv.\n- Hun blev farlig"}, {"TEXT": "Hun blev farlig til sidst, når hun begyndte at ramme linjerne og ramme hjørnerne. Så var det ikke nemt at returnere, siger Wozniacki"}]}' 
  actual_QUOTES_r = dataset[idx]['rejected']

  assert actual_QUOTES_r == EXP_QUOTES_r, f"QUOTES filled check - rejected: Actual QUOTES: <{actual_QUOTES_r}> does not match expected QUOTES: <{EXP_QUOTES_r}>"

## QUOTES check: chosen (article does NOT contain quotes) ##
  idx = 30
  EXP_QUOTES_c =  '{"QUOTES": []}'
  actual_QUOTES_c = dataset[idx]['chosen']

  assert actual_QUOTES_c == EXP_QUOTES_c, f"QUOTES empty check - chosen: Actual QUOTES: <{actual_QUOTES_c}> does not match expected QUOTES: <{EXP_QUOTES_c}>"

## QUOTES check: rejected
  EXP_QUOTES_r =  '{"QUOTES": [{"TEXT": "Den danske badmintonspiller Viktor Axelsen er klar til semifinalen ved EM i badminton efter en sejr i to sæt over Thomas Rouxel fra Frankrig."}, {"TEXT": "Den danske singlespiller Line Kjærsfeldt er også færdig ved EM, efter hun skulle have mødt Carolina Marin i kvartfinalen."}]}'
  actual_QUOTES_r = dataset[idx]['rejected']

  assert actual_QUOTES_r == EXP_QUOTES_r, f"QUOTES empty check - rejected: Actual QUOTES: <{actual_QUOTES_r}> does not match expected QUOTES: <{EXP_QUOTES_r}>"

## SRC check: chosen ####################################
  idx = 4
  EXP_SRC_c = '{"SRC": ["Caroline Wozniacki"]}'
  actual_SRC_c = dataset[idx]['chosen']

  assert actual_SRC_c == EXP_SRC_c, f"SRC check - chosen: Actual SRC: <{actual_SRC_c}> does not match expected SRC: <{EXP_SRC_c}>"

## SRC check: rejected
  EXP_SRC_r = '{"SRC": []}'
  actual_SRC_r = dataset[idx]['rejected']

  assert actual_SRC_r == EXP_SRC_r, f"SRC check - rejected: Actual SRC: {actual_SRC_r} does not match expected rejected SRC: <{EXP_SRC_r}>"

## GEN check: chosen ####################################
  idx = 11
  EXP_GEN_c = '{"GEN": "F"}'
  actual_GEN_c = dataset[idx]['chosen']

  assert actual_GEN_c == EXP_GEN_c, f"GEN check - chosen: Actual GEN: <{actual_GEN_c}> does not match expected GEN: <{EXP_GEN_c}>"

## GEN check: rejected
  EXP_GEN_r = '{"GEN": "M"}'
  actual_GEN_r = dataset[idx]['rejected']

  assert actual_GEN_r == EXP_GEN_r, f"GEN check - rejected: Actual GEN: {actual_GEN_r} does not match expected rejected GEN: <{EXP_GEN_r}>"

## FUN check: chosen ####################################
  idx = 13
  EXP_FUN_c = '{"FUN": "Other"}'
  actual_FUN_c = dataset[idx]['chosen']

  assert actual_FUN_c == EXP_FUN_c, f"FUN check - chosen: Actual FUN: <{actual_FUN_c}> does not match expected FUN: <{EXP_FUN_c}>"

## FUN check: rejected
  EXP_FUN_r = '{"FUN": "Interest organization"}'
  actual_FUN_r = dataset[idx]['rejected']

  assert actual_FUN_r == EXP_FUN_r, f"FUN check - rejected: Actual FUN: {actual_FUN_r} does not match expected rejected FUN: <{EXP_FUN_r}>"

## EMP check: chosen ####################################
  idx = 20
  EXP_EMP_c = '{"EMP": []}'
  actual_EMP_c = dataset[idx]['chosen']

  assert actual_EMP_c == EXP_EMP_c, f"EMP check - chosen: Actual EMP: <{actual_EMP_c}> does not match expected EMP: <{EXP_EMP_c}>"

## EMP check: rejected
  EXP_EMP_r = '{"EMP": ["Kommunernes Landsforening"]}'
  actual_EMP_r = dataset[idx]['rejected']

  assert actual_EMP_r == EXP_EMP_r, f"EMP check - rejected: Actual EMP: {actual_EMP_r} does not match expected rejected EMP: <{EXP_EMP_r}>"

## ROL check: chosen ####################################
  idx = 24
  EXP_ROL_c = '{"ROL": ["tennisstjerne"]}'
  actual_ROL_c = dataset[idx]['chosen']

  assert actual_ROL_c == EXP_ROL_c, f"ROL check - chosen: Actual ROL: <{actual_ROL_c}> does not match expected ROL: <{EXP_ROL_c}>"

## ROL check: rejected
  EXP_ROL_r = '{"ROL": ["far", "racerkører"]}'
  actual_ROL_r = dataset[idx]['rejected']

  assert actual_ROL_r == EXP_ROL_r, f"ROL check - rejected: Actual ROL: {actual_ROL_r} does not match expected rejected ROL: <{EXP_ROL_r}>"



