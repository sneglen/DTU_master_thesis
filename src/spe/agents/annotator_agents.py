# standard libraries
import logging
import json
from typing import List, Dict, Any, Optional

# third-party libraries
import demjson
import sglang as sgl
from omegaconf import DictConfig

# local libraries
from src.spe.annotator_instructions import get_instructions
import src.spe.agents.pydantic_annotator_classes as pac 

logger = logging.getLogger(__name__)

TEMPERATURE = 0.0 # Bug in @sgl.function which does not allow to pass temperature

def parse_json_with_fallback(json_str):

  json_dict = None
  parse_ok = True

  try:
    # Naive approach: try to load the JSON string
    json_dict =  json.loads(json_str)

  except json.JSONDecodeError:
    # Standard JSON parsing failed, try with demjson
    logger.notice("JSON string needs correction.")

    try:
      json_dict = demjson.decode(json_str, encoding='utf8')

    except demjson.JSONDecodeError:
      logger.notice(f"Error: JSON string correction failed: {json_str}. Returning default dict.")
      parse_ok = False

  return (json_dict, parse_ok)


def check_for_hallucinations(json_str_response, key):
  hallucination_detected = False
  if key in ["PER", "ORG"]:
    if len(json_str_response) > 200:
      hallucination_detected = True
  elif key in ["SRC", "ROL", "EMP"]:
    if len(json_str_response) > 100:
      hallucination_detected = True
  elif key in ["GEN"]:
    if len(json_str_response) > 20:
      hallucination_detected = True
  elif key in ["FUN"]:
    if len(json_str_response) > 40:
      hallucination_detected = True
  elif key in ["QUOTES"]:
    # check if it contains more that 25 consecutives spaces
    if ' ' * 25 in json_str_response:
      hallucination_detected = True

  if hallucination_detected:
    logger.notice(f"Warning: {key} hallucination detected.")


def src_entity_found_in_org(src_dict, org_dict):
    for src_item in src_dict['SRC']: # (there could be multiple SRCs in a quote)
      if src_item in org_dict['ORG']:
        return True
    return False

##############################################################################################################
##############################################################################################################

@sgl.function
def worker_with_regex(s, worker_name: str, max_tokens: int, regex_input:str, instructions: str):
  
  s += instructions + "\n"
  s += sgl.gen(name=worker_name, max_tokens=max_tokens, temperature=TEMPERATURE, regex=regex_input)


###################################################
def general_annotator(cfg: DictConfig, text: str, key: str, quote_text: Optional[str] = None, per_json_dict: Optional[Dict[str, Any]] = None, org_json_dict: Optional[Dict[str, Any]] = None, src_json_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  """To annotate PER+ORG (and find QUOTES), the optional arguments are None.
    To annotate QUOTES, the optional arguments (quote_text, per_json_dict, org_json_dict) are REQUIRED."""

  logger.notice(f"Annotating {key}:")
  raw_response = 'KEY:<{key}>RAWBEGIN:<nothing retrieved>:RAWEND\n'

  default_dict_for_key = {key: pac.get_default_dict_for_key(key)}

  ## VALIDATE INPUTS: Return the corresponding default dict for the annotator key in case of wrong input arguments

  # {PER, ORG, QUOTE} dicts only needed to generate instructions to annotate the QUOTES
  if key in ["PER", "ORG", "QUOTES"]:
    if quote_text is not None or per_json_dict is not None or org_json_dict is not None:
      logger.error(f"Error: Optional arguments must be None for {key} annotator.")
      return default_dict_for_key, raw_response

  # check that optional arguments are NOT None if key is a QUOTE-key
  # they are needed to generate instructions for the QUOTES annotator
  elif key in ["SRC", "GEN", "FUN", "ROL", "EMP"]:
    if quote_text is None or per_json_dict is None or org_json_dict is None:
      logger.error(f"Error: Optional arguments must NOT be None for {key} annotator.")
      return default_dict_for_key, raw_response
        
  else:
    logger.error(f"Error: '{key}' is not a valid annotator key.")
    # Problem cannot be solved with default_dict_for_key. Must return None.
    return None, raw_response


  ## MAIN logic
  default_json_str_response = json.dumps(default_dict_for_key)
  json_str_response = default_json_str_response 
  
  regex_map = {
    "PER": pac.per_regex,
    "ORG": pac.org_regex,
    "QUOTES": pac.quotes_text_regex,
    "SRC": pac.src_regex,
    "GEN": pac.gen_regex,
    "FUN": pac.fun_regex,
    "ROL": pac.rol_regex,
    "EMP": pac.emp_regex
    }

  if key in regex_map:
    regex_input = regex_map[key]    
    instructions = get_instructions(key=key, article_text=text, quote_text=quote_text, per_dict=per_json_dict, org_dict=org_json_dict, src_dict=src_json_dict)

    if instructions is None:
      logger.notice(f"{key} annotation error: No instructions retrieved. Returning default dict.")
      return default_dict_for_key, raw_response

    else:
      try:
        state = worker_with_regex.run(worker_name="response", max_tokens=1000, regex_input=regex_input, instructions=instructions) 
        try:
          # If the response is empty, return the default response
          # (to avoid errors afterwards)
          if state['response'] == "{}" or state['response'] == "{ }" or state['response'] == "":
            logger.notice(f'WARNING: Empty response for <{key}>. Returning default dict.')
            json_str_response = default_json_str_response
          else:
            json_str_response = state['response']
            raw_response = f'KEY:<{key}>RAWBEGIN:{json_str_response}:RAWEND\n'

        except Exception as e:   
          logger.notice(f"Could not get response for: {key}. Error catched: {e}")
          json_str_response = default_json_str_response # default response if worker response is missing

      except Exception as e:
        logger.notice(f"{key} annotation error: catched in general_annotator() while calling worker_with_regex.run(): {e}. Returning default dict.")
        return default_dict_for_key, raw_response

  else:
    logger.error(f"Error: '{key}' is not a valid regex key. Cannot return default dict.")
    return None, raw_response

  check_for_hallucinations(json_str_response, key)
  (json_dict, parse_ok) = parse_json_with_fallback(json_str_response)

  if not parse_ok:
    logger.error(f"Error: Failed to correct {key} annotator response. Continue annotation with default dict for {key}.")
    json_dict = default_dict_for_key

  # Problem-free return (response retrieved without errors)
  return json_dict, raw_response


##############################################################################################################
##############################################################################################################
def quote_annotator(cfg: DictConfig, article, quotes_list: List[Dict[str, Any]], per_dict: dict, org_dict: dict):
  # Initialize default quote category structures from hydras config

  article_text = article["TEXT"]

  quote_keys = {"SRC", "GEN", "FUN", "ROL", "EMP"}
  annotated_quotes = []
  total_raw_response = ''

  default_quote_unit_dict = {key: pac.get_default_dict_for_key(key) for key in cfg.spe.annotate if key in quote_keys}

  for idx, quote in enumerate(quotes_list):

    msg = f"QUOTE [{idx}/{len(quotes_list)-1}]: '{quote['TEXT'][:60]}...'"
    logger.notice(msg + "\n" + "-" * 78)

    src_dict = None

    # Add 'TEXT' key before the other keys
    quote_unit_dict = {'TEXT': quote['TEXT']}
    quote_unit_dict.update(default_quote_unit_dict.copy())

    for key in quote_unit_dict.keys():
      try:
        if key != 'TEXT' and cfg.spe.annotate[key]:
          if cfg.spe.annotate_bypass.get(key, False): # bypass notation (default false, just in case)
            annotated_dict = {key: quote[key]}
          else:  
            # EXCLUDE annotation for certain cases:
            if key == "SRC" and not per_dict['PER']:
              # If per_dict is empty then SRC must be empty (default).
              annotated_dict = {key: pac.get_default_dict_for_key(key)}
            elif key == "GEN" and not per_dict['PER']:
              # If per_dict is empty then GEN is 'X' (default).
              annotated_dict = {key: pac.get_default_dict_for_key(key)}
            elif key == "FUN" and src_entity_found_in_org(src_dict, org_dict):
              # if SRC is an ORG, then FUN is 'X' (default).
              annotated_dict = {key: pac.get_default_dict_for_key(key)}
            else:  
              try:
                annotated_dict, raw_response = general_annotator(cfg, article_text, key=key, quote_text=quote["TEXT"], per_json_dict=per_dict,  org_json_dict=org_dict, src_json_dict=src_dict)
                total_raw_response += raw_response
              except Exception as e:
                logger.error(f"Returning default dict: Annotation was not excluded but triggered an error in general_annotator(key={key}) for quote [{idx}]: {e}")
                annotated_dict = {key: pac.get_default_dict_for_key(key)}
                total_raw_response += f"KEY:<{key}>RAWBEGIN:<nothing retrieved>:RAWEND\n"

          if key == "SRC": 
            # After annotating SRC, pass <src_dict> to the subsequent general_annotator() calls to annotate GEN, FUN, ROL, EMP
            src_dict = annotated_dict      
          quote_unit_dict[key] = annotated_dict[key]
      except Exception as e:
        logger.error(f"Error in key for-loop annotating <{key}> for quote [{idx}]: {e}")
        quote_unit_dict[key] = pac.get_default_dict_for_key(key)

    annotated_quotes.append(quote_unit_dict)

  return annotated_quotes, total_raw_response