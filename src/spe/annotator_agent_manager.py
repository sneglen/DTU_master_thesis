# standard libraries
import logging

# third-party libraries
from omegaconf import DictConfig

# local libraries
from src.spe.agents import agent

logger = logging.getLogger(__name__)


def annotator_agent_manager(cfg: DictConfig, article):
    """Annotates the target article using the annotator agent."""

    article_text = article["TEXT"]

    json_dict_annotator_response = {}
    total_raw_response = ''
    
    # Find PER
    # Expected output: per_json_dict  =  {'PER': ['Peter', 'Susanne Jensen']}
    try:
      if cfg.spe.annotate.PER:
        if cfg.spe.annotate_bypass.PER:
          per_json_dict = {'PER': article['PER']}
        else:
          per_json_dict, raw_response = agent.general_annotator(cfg, article_text, key="PER")
          total_raw_response += raw_response
        
        json_dict_annotator_response.update(per_json_dict)

      #logger.notice(f'per_json_dict: {per_json_dict}')        
    except Exception as e:
      logger.error(f'Error in annotating PER (returning empty response): {e}')
      json_dict_annotator_response.update({'PER': []})

    # Find ORG
    # Expected output: org_json_dict = {'ORG': ['Instagram', 'Bilka']}
    try:
      if cfg.spe.annotate.ORG:
        if cfg.spe.annotate_bypass.ORG:
          org_json_dict = {'ORG': article['ORG']}
        else:
          org_json_dict, raw_response = agent.general_annotator(cfg, article_text, key="ORG")
          total_raw_response += raw_response

        json_dict_annotator_response.update(org_json_dict)
      
      #logger.notice(f'org_json_dict: {org_json_dict}')        
    except Exception as e:
      logger.error(f'Error in annotating ORG (returning empty response): {e}')
      json_dict_annotator_response.update({'ORG': []})

    # Find QUOTES
    # Expected output:
    # quotes_json_dict = {'QUOTES': [
    # {'TEXT': 'this is a quote', 
    #  'GEN': 'F', 
    #  'FUN': 'Other', 
    #  'SRC': ['Caroline Wozniacki'], 
    #  'ROL': ['tennisstjerne'], 
    #  'EMP': []},
    # {...},
    # {...}
    # ]}
    try:
      if cfg.spe.annotate.QUOTES and not cfg.spe.annotate_bypass.QUOTES:
        quotes_json_dict, raw_response = agent.general_annotator(cfg, article_text, key="QUOTES")
        total_raw_response += raw_response        
      else:
        # DO NOTE this is not a leakage from target article quotes to predicted article quotes.
        # It is deliberately created to allow annotation of the remaining categories
        # when finding QUOTES is bypassed. In the common scenario, no category is bypassed
        # an hence this code is not executed (no leakage from target data).

        # strip the quotes from the article
        quotes_json_dict = [{'TEXT': quote['TEXT']} for quote in article['QUOTES']]
        quotes_json_dict = {'QUOTES': article['QUOTES']}
        
    except Exception as e:
      logger.error(f'Error in finding QUOTES (returning empty response): {e}')
      quotes_json_dict = {'QUOTES': []}

    # Annotate QUOTES
    # 'QUOTES' is added in case it is bypassed, but SRC, GEN, FUN, ROL, EMP are not to be annotated
    # so a valid quotes_json_dict is created (with 'TEXT' and the rest of the keys set to default)
    try:
      if any(cfg.spe.annotate[key] for key in ['QUOTES', 'SRC', 'GEN', 'FUN', 'ROL', 'EMP']):
        quotes_annotated_dict, raw_response = agent.quote_annotator(cfg, article, quotes_json_dict['QUOTES'], per_json_dict, org_json_dict)
        total_raw_response += raw_response
        # Replace found quotes with annotated versions
        json_dict_annotator_response['QUOTES'] = quotes_annotated_dict 
    except Exception as e:
      logger.error(f'Error calling agent.quote_annotator() (returning empty response): {e}')
      json_dict_annotator_response.update({'QUOTES': []})

    return json_dict_annotator_response, total_raw_response