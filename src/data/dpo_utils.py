import os
import logging
import pickle
import random
import string

# third-party libraries
from fuzzywuzzy import fuzz, process
from omegaconf import DictConfig

# local libraries
from src.config import logging_config
import src.utils.hydra_helper as hh
import src.data.data_utils as du
import src.data.metrics as mt


logger = logging.getLogger(__name__)


def generate_rejected_quote_text_dict(article_txt, quote_txt):

    ch_offset = 20
    # Find closest match in article
    #best_match = process.extractOne(quote_txt, 
    #                                [article_txt[i:i+len(quote_txt)+ch_offset] for i in range(len(article_txt) - len(quote_txt) + 1)], 
    #                                scorer=fuzz.partial_ratio)
    
    # The quote may not be exact in the article, therefore some heuristic is needed
    #article_txt = article_txt.replace('\n', '')
    #quote_txt = quote_txt.replace('\n', '')

    best_quote_matches = process.extractOne(quote_txt, 
                                    [article_txt[i:i+len(quote_txt)] for i in range(len(article_txt) - len(quote_txt) + 1)], 
                                    scorer=fuzz.UWRatio)
    
    best_quote_txt = best_quote_matches[0]
    quote_idx_in_article = article_txt.find(best_quote_txt)

    
    if quote_idx_in_article is None:
      logger.notice("Quote not found in the article. Returning empty quote as rejected.")
      return {'TEXT': ''}
        
    fetching_options = ['chop']

    # Case 1: Quote too close to the beginning of the article.
    #         try to append succeeding text from article.
    #         else return half of the quote.
    if len(best_quote_txt) + ch_offset < len(article_txt) - quote_idx_in_article:
      # do append 3 times (better simulate the wrong tendency of the LLM to append text)
      fetching_options.append('append')
      fetching_options.append('append')
      fetching_options.append('append')
        
    # Case 2: Quote is not within the first ch_offset characters in the article. 
    #         try to append preceeding text from article.
    #         else return half of the quote.
    if len(article_txt) - quote_idx_in_article > len(best_quote_txt):
      fetching_options.append('preceed')

    selected_option = random.choice(fetching_options)

    chars_to_strip = string.punctuation + string.whitespace

    if selected_option == 'append':
      raw_quote_txt = article_txt[quote_idx_in_article : quote_idx_in_article + len(best_quote_txt) + ch_offset]
 
    elif selected_option == 'preceed':
      raw_quote_txt = article_txt[quote_idx_in_article - ch_offset : quote_idx_in_article + len(best_quote_txt)]
    
    else: # chop it (as default/fallback option)
      half_idx = len(best_quote_txt) // 2
      raw_quote_txt = best_quote_txt[:half_idx]

    return_quote_txt = raw_quote_txt.strip(chars_to_strip)

    return {'TEXT': return_quote_txt}


def get_random_entries_from_file(cfg, key, target_list):

  if not isinstance(target_list, list):
    logger.error(f'<target_list> for <{key}> must be a list.')

  try:  
    max_retrieval = cfg.data.dpo[key].max_retrieval
  except Exception as e:
    logger.error(f'Error reading max_retrieval for key <{key}>: {e}')

  try:
    with open(cfg.data.dpo[key].file, 'r', encoding='utf-8') as file:
      entries = file.read().splitlines()
  except Exception as e:
    logger.error(f'Error reading file entries for key <{key}>: {e}')

  
  # Filter out entries that are in the target_list
  # (but first verify it is not None)
  if target_list is None:
    logger.error(f'Error: target_list for <{key}> is None.')

  entries = [entry for entry in entries if entry not in target_list]

  if not entries:
    return {key: []}

  min_number = 0
  # Be sure not to return empty list if target_list is empty
  # (Many different ways to check if list is empty, but this is the most robust one)
  if target_list == [] or target_list == [''] or len(target_list) == 0:
    min_number = 1

  num_entries = random.randint(min_number, max_retrieval)

  selected_entries = random.sample(entries, min(num_entries, len(entries)))

  # Repeat one entry if the selected don't reach max_retrieval
  # Sometimes the LLM may repeat the same entry

  # Do not repeat empty entries
  if selected_entries != [] and len(selected_entries) >= cfg.data.dpo.allow_repeat_at and len(selected_entries) < max_retrieval:
      repeated_entry = random.choice(selected_entries)
      selected_entries.append(repeated_entry)

  return {key: selected_entries}


def generate_rejected_dict(cfg, key, evaluation, target_unit, predicted_unit):

  # <predicted_unit> can be an article or a quote with its corresponding <evaluation>

  # <predicted_unit> = article:
  # For PER+ORG, the evaluation is at "article level":
  # eg. evaluation['PER'].all_rejected()

  # <predicted_unit> = quote:
  # For QUOTES (TEXT,GEN,FUN,SRC,ROL.EMP), the evaluation is at "quote level":
  # eg. evaluation['QUOTES'][quote_idx]['SRC'].all_rejected() 
  # but because the quote evaluation is passed, then the same syntax as for PER+ORG can be used:
  # eg. evaluation['SRC'].all_rejected()
   
  list_of_choices = None
  if key == 'GEN':
    list_of_choices = ['M', 'F', 'X']
  elif key == 'FUN':
    list_of_choices = ['Expert', 'Case', 'Politician', 'DR source', 'Interest organization', 'Professional expert', 'Authority', 'Other']

  # Important to test first that evaluation is not None (otherwhise .all_rejected() will throw an error)
  # If all predictions are wrong, the predictions can be returned as rejected
  if evaluation and predicted_unit and evaluation[key].all_rejected():
    # return predicted dict: it is already rejected
    rejected_dict = {key: predicted_unit[key]}
  else:
    if key in ['GEN', 'FUN']:
      # random choice between the remaining possibilities
      remaining_choices = [choice for choice in list_of_choices if choice != target_unit[key]]
      selected_choice = random.choice(remaining_choices)
      rejected_dict = {key: selected_choice}
    
    elif key in ['PER', 'ORG', 'SRC', 'ROL', 'EMP']:
      rejected_dict = get_random_entries_from_file(cfg, key=key, target_list=target_unit[key])

    else:
      logger.error(f"Error: Key <{key}> not recognized.")   

  return rejected_dict

def generate_single_quote_dict(cfg, target_article, target_quote_idx, predicted_quote=None, quote_eval=None):

  if predicted_quote:
    quote_state = 'existing'
  else:
    quote_state = 'missing'

  logger.notice(f"Generating <{quote_state}> quote[{target_quote_idx}/{len(target_article['QUOTES'])-1}]")
  
  target_quote = target_article['QUOTES'][target_quote_idx]

  rejected_quote_dict = {}

  rejected_TEXT = generate_rejected_quote_text_dict(target_article['TEXT'], target_quote['TEXT'])
  rejected_quote_dict.update(rejected_TEXT)

  keys = ['GEN', 'FUN', 'SRC', 'ROL', 'EMP']

  for key in keys: 
    rejected_dict = generate_rejected_dict(cfg              = cfg, 
                                            key             = key, 
                                            evaluation      = quote_eval, 
                                            target_unit     = target_quote,
                                            predicted_unit  = predicted_quote)    
    rejected_quote_dict.update(rejected_dict)

  return rejected_quote_dict


def generate_all_quotes(cfg, article_evaluation, target_article, predicted_article):

  list_of_quotes = []
  # CASE 1 (error): Number of evaluated targets and actual targets must match
  if article_evaluation and article_evaluation['TEXT'].n_targets != len(target_article['QUOTES']):
    breakpoint()
    logger.error("Error: Number of target quotes and number of target quotes in article_evaluation do not match.")

  # CASE 2 (no target quotes): return empty quote list
  elif len(target_article['QUOTES']) == 0:
    logger.notice('Generating quotes skipped: Article has no quotes')

  # CASE 3a (target quotes but missing evaluation): generate rejected quotes from scratch
  # CASE 3b (target quotes but missing predicted article): generate rejected quotes from scratch
  elif (article_evaluation is None or predicted_article is None): 
    for target_quote_idx in range(len(target_article['QUOTES'])):
      rejected_quote_dict = generate_single_quote_dict(cfg, target_article, target_quote_idx)
      list_of_quotes.append(rejected_quote_dict)

  # CASE 4: There are quotes and the evaluation is present: generate rejected quotes based on evaluation
  else:
    # Iterate through target quotes (some may not have a matching predicted quote)
    for target_quote_idx in range(article_evaluation['TEXT'].n_targets):
      logger.notice(f"\nTarget quote [{target_quote_idx}/{article_evaluation['TEXT'].n_targets-1}]")

      # Find possible matching predicted quote for target_quote_idx of the above for-loop 
      matching_predicted_quote_idx = None
      matching_eval_quote_idx      = None
      for eval_quote_idx in range(article_evaluation['TEXT'].n_matches):
        if article_evaluation['QUOTES'][eval_quote_idx]['DRidx']['t'] == target_quote_idx:
          matching_predicted_quote_idx = article_evaluation['QUOTES'][eval_quote_idx]['DRidx']['p']
          matching_eval_quote_idx = eval_quote_idx
          break 
      
      # A) No matching predicted quote found: generate missing rejected quote
      if matching_predicted_quote_idx is None:
        #logger.notice(f"Target quote[{target_quote_idx}] has no matching predicted quote. Generate missing rejected quote.")
        rejected_quote_dict = generate_single_quote_dict(cfg, target_article, target_quote_idx)

      # B) Matching predicted quote found: go inside quote and generate rejected data
      else:
        logger.notice(f"Target quote[{target_quote_idx}] -> Predicted quote[{matching_predicted_quote_idx}]")
        quote_eval = article_evaluation['QUOTES'][matching_eval_quote_idx]
        predicted_quote = predicted_article['QUOTES'][matching_predicted_quote_idx]
        rejected_quote_dict = generate_single_quote_dict(cfg, target_article, target_quote_idx, predicted_quote, quote_eval)

      list_of_quotes.append(rejected_quote_dict)


  rejected_quotes_dict = {'QUOTES':  list_of_quotes}


  return rejected_quotes_dict


def generate_rejected_dpo_data(cfg: DictConfig):
  logger = logging.getLogger(logging_config.logger_name)

  data_dir = hh.get_data_dir(cfg)
  logger.notice(f"Data dir: {data_dir}")

  random.seed(cfg.random_seed)

  wuzzy_conf = cfg.data.wuzzy_conf

  file_forename = "article_"    
  base_filepath = os.path.join(data_dir, f"{file_forename}")
  DR_indices, missing_p_indices = mt.get_DR_indices_from_folder(data_dir, file_forename)

  # Include missing predicted articles (so rejected entries can be generated for them as well)
  DR_indices = sorted(DR_indices + missing_p_indices) 

  rejected_articles = []

  if not DR_indices:
    logger.notice(f"No articles found to process: <{data_dir}>.")
    return
  else:
    for DR_idx in DR_indices:
      try:
        base_filepath_idx = base_filepath + str(DR_idx)
  
        # Load target article
        target_article = du.load_article_from_json(base_filepath_idx + "t.json")
        if target_article is None:
          logger.error(f"Error: Target article not found: {base_filepath_idx}t.json")
          continue        

        logger.notice(f"\nGenerated rejected art.[{DR_idx}]: {target_article['headline'][:60]}...")
          
        # Load predicted article
        predicted_article = du.load_article_from_json(base_filepath_idx + "p.json")

        if predicted_article:
          article_evaluation = mt.evaluate_article(target_article, predicted_article, wuzzy_conf)
        else:
          logger.notice(f"Predicted article not found: {base_filepath_idx}p.json")
          article_evaluation = None

        # Include "header" information in article
        rejected_article = {}
        rejected_article.update({'headline': target_article['headline']})
        rejected_article.update({'url': target_article['url']})
        rejected_article.update({'TEXT': target_article['TEXT']})

        # Generate rejected article categories: PER, ORG, QUOTES
        rejected_PER = generate_rejected_dict(cfg             = cfg, 
                                              key             = 'PER', 
                                              evaluation      = article_evaluation, 
                                              target_unit     = target_article,
                                              predicted_unit  = predicted_article)                
        rejected_article.update(rejected_PER)

        rejected_ORG = generate_rejected_dict(cfg             = cfg, 
                                              key             = 'ORG', 
                                              evaluation      = article_evaluation, 
                                              target_unit     = target_article,
                                              predicted_unit  = predicted_article)
        rejected_article.update(rejected_ORG)

        rejected_QUOTES = generate_all_quotes(cfg, article_evaluation, target_article, predicted_article)
        rejected_article.update(rejected_QUOTES)
        
        rejected_articles.append(rejected_article)

      except Exception as e:
          logger.error(f"Error generating rejected article[{DR_idx}]: {e}")
          continue
      
    # Save pickle dataset
    try:
      with open(cfg.data.split.dpo, 'wb') as file:
        pickle.dump(rejected_articles, file)
        logger.notice(f'dpo dataset saved: {cfg.data.split.dpo}')
    except Exception as e:
      logger.error(f"Error saving dpo pickle dataset: {e}")