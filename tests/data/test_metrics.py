# standard libraries
import numpy as np
import numpy.testing as npt
from unittest.mock import patch
import json

# third-party libraries
import pytest as pt
from fuzzywuzzy import fuzz

# local libraries
from src.data.metrics import evaluate_entity_list
from src.data.metrics import calculate_quote_similarity_matrix
from src.data.metrics import find_matching_quotes
from src.data.metrics import evaluate_quotes
from src.data.metrics import evaluate_article
from src.data.metrics import EvaluationData

## Fixtures and helper functions ############################################################################
@pt.fixture
def FIX_matching_quotes_articles():
  return {
  "target_article": {"QUOTES": [
        {"TEXT": "Peter og Sofus arbejder i børsen"},   # <--
        {"TEXT": "han er en flittig mand"},             # <--
        {"TEXT": "Jens plejer at spise is"},
        {"TEXT": "Dette citat er ikke fundet"},
        {"TEXT": "Dette citat er heller ikke fundet"}]
        }, 
  "predicted_article": {"QUOTES": [
        {"TEXT": "han er en flittig mand sagde Peter"}, # <-- acceptable match
        {"TEXT": "Peter og Sofus arbejder i børsen"},   # <-- perfect match
        {"TEXT": "Jens bliver ved med at spise is"},    #  x  bad match
        {"TEXT": "Dette citat eksisterer ikke"}]}       #  x  spurious quote  
  }

def fcn_brief_wuzzy_conf(mode: str = "strict", threshold: int = 100, ratio_fcn: str = "fuzz.WRatio") -> dict:
  return {
    "TEXT": {"threshold": threshold, "ratio_fcn": ratio_fcn},
    # Parameters for all other entities types than 'TEXT'
    "mode": mode,
    "threshold": threshold 
  }

def fcn_eval_quotes_wuzzy_conf():
  return {
      'PER': {'mode': 'strict', 'threshold': 100},
      'ORG': {'mode': 'strict', 'threshold': 100},
      'TEXT': {'ratio_fcn': 'fuzz.WRatio', 'threshold': 100},
      'GEN': {'mode': 'strict'},
      'FUN': {'mode': 'relaxed'},
      'SRC': {'mode': 'relaxed', 'threshold': 1}, # pred "Jens Petar" would be a match for "Jens" with threshold 1
                                                  # but because "Jens" in target is excluded because there is already a better match
                                                  # then "Jens Petar" is not considered a match for "Jens" in the evaluation
      'ROL': {'mode': 'relaxed', 'threshold': 90},
      'EMP': {'mode': 'relaxed', 'threshold': 90}
	}


## evaluate_entity_list #############################################################################
## Data
tar_filled = ["victor Jespersen", "PETER", "elias", "eliot", "sofus", "alvin"]
pre_filled = ["olivia", "PETER","victor Jesper"   , "sofAs", "elLiotT"]
#                                        ***       *       *   *   <--- Intentional errors

# empty target or prediction considered as an entry so e.g. both empty gives: 1 match, 1 target and 1 prediction
tar_empty = [""] 
pre_empty = [""]

# Ensure GEN+FUN can be handled as well: They are just a single string (no a list)
tar_FUN = "Professional expert" 

@pt.mark.parametrize("test_data,                           wuzzy_conf, EXP_n_matches, EXP_n_targets, EXP_n_predictions", [
      ((tar_filled, pre_filled), fcn_brief_wuzzy_conf("strict",  100),             1,             6,                 5),  
      ((tar_filled, pre_filled), fcn_brief_wuzzy_conf("relaxed",  80),             4,             6,                 5),
       ((tar_filled, pre_empty), fcn_brief_wuzzy_conf("strict",  100),             0,             6,                 1),
       ((tar_empty, pre_filled), fcn_brief_wuzzy_conf("strict",  100),             0,             1,                 5),
        ((tar_empty, pre_empty), fcn_brief_wuzzy_conf("strict",  100),             1,             1,                 1),
         ((tar_FUN, pre_empty),  fcn_brief_wuzzy_conf("strict",  100),             0,             1,                 1),
  ])
def test_evaluate_entity_list(logger, request, test_data, wuzzy_conf, EXP_n_matches, EXP_n_targets, EXP_n_predictions):
  logger.info(f"TEST: {request.node.name}")

  (target_list, predicted_list) = test_data


## Execute test  
  res = evaluate_entity_list(target_list, predicted_list, wuzzy_conf, "dummy entity key")

## Assert results
  assert res.n_matches == EXP_n_matches, f"Expected total matches: {EXP_n_matches}, got: {res.n_matches}"
  assert res.n_targets == EXP_n_targets, f"Expected total targets: {EXP_n_targets}, got: {res.n_targets}"
  assert res.n_predictions == EXP_n_predictions, f"Expected total predictions: {EXP_n_predictions}, got: {res.n_predictions}"


## calculate_quote_similarity_matrix #############################################################################
# (columns: 4 predicted, rows: 5 targets)
EXP_res_WR = np.array([[41, 100,  40,  45],\
                       [90,  30,  38,  33],\
                       [40,  46,  78,  48],\
                       [45,  39,  37,  72],\
                       [46,  41,  38,  70]])

# (columns: 4 predicted, rows: 5 targets)
EXP_res_pr = np.array([[35,  100,  37,  41],\
                       [100,  36,  41,  32],\
                       [43,   47,  70,  48],\
                       [42,   38,  38,  69],\
                       [39,   41,  40,  78]])
@pt.mark.parametrize("ratio_fcn, EXP_results", [(fuzz.WRatio,        EXP_res_WR), 
                                                (fuzz.partial_ratio, EXP_res_pr)])
def test_calculate_quote_similarity_matrix(logger, request, ratio_fcn, EXP_results, FIX_matching_quotes_articles):
  logger.info(f"TEST: {request.node.name}")
  target_article = FIX_matching_quotes_articles["target_article"]
  predicted_article = FIX_matching_quotes_articles["predicted_article"]

## Execute test
  res = calculate_quote_similarity_matrix(target_article, predicted_article, ratio_fcn)

## Assert results 
  assert res.shape == EXP_results.shape, f"Expected shape {EXP_results.shape}, got {res.shape}"
  npt.assert_array_almost_equal(res, EXP_results, decimal=0)


## find_matching_quotes #############################################################################
def test_find_matching_quotes(logger, request, FIX_matching_quotes_articles):
  logger.info(f"TEST: {request.node.name}")
  # (columns: 4 predicted, rows: 5 targets)
  mocked_similarity_matrix = np.array([[35,  99,  37,  41],\
                                      [100,  36,  41,  32],\
                                      [43,   47,  70,  48],\
                                      [42,   38,  38,  69],\
                                      [39,   41,  40,  78]])

  EXP_res = [((1, 0), 100), ((0, 1), 99), ((4, 3), 78)]
  EXP_res_sorted = sorted(EXP_res, key=lambda x: (x[0], x[1]))

  target_article = FIX_matching_quotes_articles["target_article"]
  predicted_article = FIX_matching_quotes_articles["predicted_article"]

## Execute test  
  with patch('src.data.metrics.calculate_quote_similarity_matrix', return_value=mocked_similarity_matrix):
      res = find_matching_quotes(target_article, predicted_article, fcn_brief_wuzzy_conf(threshold=75))
  
  r_sorted = sorted(res, key=lambda x: (x[0], x[1]))


## Assert results
  assert len(r_sorted) == len(EXP_res), f"Expected shape {len(EXP_res)}, got {len(r_sorted)}"
  for actual, expected in zip(r_sorted, EXP_res_sorted):
        assert actual == expected, f"Result {actual} does not match expected {expected}."  



## evaluate_quotes #############################################################################
@pt.fixture
def FIX_evaluating_quotes_articles():
  return {
  "target_article": {
      "PER": ["Peter Lange", "Sofus", "Jens"],
      "ORG": ["IKEA", "Rinkøbing Bank", "Børsen"],
      "QUOTES": [
        {"TEXT": "Peter og Sofus arbejder i børsen", 
            "GEN": "M", 
            "FUN": "Ekspert", 
            "SRC": ["Petar", "Sofas"], 
            "ROL": "arbejder", 
            "EMP": "børsen"},   
        {"TEXT": "han er en flittig mand",
            "GEN": "M", 
            "FUN": "Ekspert", 
            "SRC": ["Petar", "Sofas"], 
            "ROL": "arbejder", 
            "EMP": "børsen"},
        {"TEXT": "Dette citat er ikke fundet",
            "GEN": "M", 
            "FUN": "Ekspert", 
            "SRC": ["Petar", "Sofas"], 
            "ROL": "arbejder", 
            "EMP": "børsen"},
        {"TEXT": "Dette citat blev noteret forkert og kan ikke bruges",
            "GEN": "", 
            "FUN": "", 
            "SRC": "", 
            "ROL": "", 
            "EMP": ""}]}, 
  "predicted_article": {
      "PER": ["Peter Lange", "Sofus", "Jens"],
      "ORG": ["IKEA", "Rinkøbing Bank", "Børsen"],
     "QUOTES": [
        {"TEXT": "han er en flittig mand sagde Peter",
            "GEN": "X", 
            "FUN": "Ekspert", 
            "SRC": ["Petar", "Sofas", "Jens Petar"], 
            "ROL": "arbejder hårdt",              # <-- acceptable match because: relaxed,90
            "EMP": "børsen"},         
        {"TEXT": "Peter og Sofus arbejder i børsen",
            "GEN": "M", 
            "FUN": ["Ekspert"], 
            "SRC": ["Petar", "Sofas"], 
            "ROL": "arbejder", 
            "EMP": "Børsen"},                     # <-- acceptable match because: relaxed,90
        {"TEXT": "Dette citat eksisterer ikke"}]} #  spurious quote, missing quote entities 
  }

def test_evaluate_quotes(logger, request, FIX_evaluating_quotes_articles):
    logger.info(f"TEST: {request.node.name}")
## Data
    EXP_res = [{
        'GEN':  EvaluationData(n_matches=1, n_targets=1, n_predictions=1),
        'FUN':  EvaluationData(n_matches=0, n_targets=1, n_predictions=1),
        'SRC':  EvaluationData(n_matches=2, n_targets=2, n_predictions=2),
        'ROL':  EvaluationData(n_matches=1, n_targets=1, n_predictions=1),
        'EMP':  EvaluationData(n_matches=1, n_targets=1, n_predictions=1),
    },{
        'GEN':  EvaluationData(n_matches=0, n_targets=1, n_predictions=1),
        'FUN':  EvaluationData(n_matches=0, n_targets=1, n_predictions=1),
        'SRC':  EvaluationData(n_matches=2, n_targets=2, n_predictions=3),
        'ROL':  EvaluationData(n_matches=1, n_targets=1, n_predictions=1),
        'EMP':  EvaluationData(n_matches=1, n_targets=1, n_predictions=1),
    }]
    EXP_res_TEXT =  EvaluationData(n_matches=2, n_targets=4, n_predictions=3)
    EXP_res_indices =  [(0, 1), (1, 0)]
     

    mocked_matching_quotes = [((0, 1), 100), ((1, 0), 100)]
 
    target_article = FIX_evaluating_quotes_articles["target_article"]
    predicted_article = FIX_evaluating_quotes_articles["predicted_article"]

## Execute test
    with patch('src.data.metrics.find_matching_quotes', return_value=mocked_matching_quotes):
      res_text, res, res_indices = evaluate_quotes(target_article, predicted_article, fcn_eval_quotes_wuzzy_conf())

## Assert results      
    # check text
    assert EXP_res_TEXT.to_tuple() == res_text.to_tuple(), f"EXP['TEXT'] = {EXP_res_TEXT.to_tuple()}, got {res_text.to_tuple()}"    

    # check quotes
    for item_idx, item in enumerate(EXP_res):
      for key in item:
        assert item[key].to_tuple() == res[item_idx][key].to_tuple(), f"EXP[{key}] = {item[key].to_tuple()}, got {res[item_idx][key].to_tuple()}"    

    # check indices
    for item_idx, item in enumerate(EXP_res_indices):
      assert item == res_indices[item_idx] , f"EXP['DRidx'] = {item}, got {res_indices[item_idx]}"    


## evaluate_article #############################################################################
def test_evaluate_articles(logger, request):
  logger.info(f"TEST: {request.node.name}")

## Data  
  EXP_res = {
    'PER':  EvaluationData(n_matches=3, n_targets=3, n_predictions=3),
    'ORG':  EvaluationData(n_matches=3, n_targets=5, n_predictions=4),
    'TEXT': EvaluationData(n_matches=4, n_targets=10, n_predictions=7),
    'GEN':  EvaluationData(n_matches=1, n_targets=4, n_predictions=4),
    'FUN':  EvaluationData(n_matches=3, n_targets=4, n_predictions=4),
    'SRC':  EvaluationData(n_matches=3, n_targets=4, n_predictions=3),
    'ROL':  EvaluationData(n_matches=3, n_targets=4, n_predictions=3),
    'EMP':  EvaluationData(n_matches=2, n_targets=4, n_predictions=4),
  }

  filename_target = "tests/data/pytest_target_article_A.json"
  filename_predicted = "tests/data/pytest_predicted_article_A.json"

  with open(filename_target, 'r', encoding='utf-8') as file:
    target_article = json.load(file)

  with open(filename_predicted, 'r', encoding='utf-8') as file:
    predicted_article = json.load(file)
  
## Execute test
  res = evaluate_article(target_article, predicted_article, fcn_eval_quotes_wuzzy_conf())

## Assert results
  for key in EXP_res:
    assert EXP_res[key].n_matches == res[key].n_matches,         f"EXP[{key}].n_matches = {EXP_res[key].n_matches}, got {res[key].n_matches}"
    assert EXP_res[key].n_targets == res[key].n_targets,         f"EXP[{key}].n_targets = {EXP_res[key].n_targets}, got {res[key].n_targets}"
    assert EXP_res[key].n_predictions == res[key].n_predictions, f"EXP[{key}].n_predictions = {EXP_res[key].n_predictions}, got {res[key].n_predictions}"  

