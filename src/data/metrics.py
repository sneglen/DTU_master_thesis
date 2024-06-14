# standard libraries
import numpy as np
import sys
import os
import re
import logging
import pickle

# third-party libraries
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from omegaconf import DictConfig

# local libraries
from src.config import logging_config
import src.utils.hydra_helper as hh
import src.data.data_utils as du
from src.data.articles_dataset import QUOTE_CATEGORIES, ARTICLE_CATEGORIES

 
# path to "type_definitions/", where QuoteCollection is defined, needed to load pickle files.
sys.path.append('src/data/dr_lib/')

logger = logging.getLogger(__name__)

# Define mapping from string identifiers to functions (from .yaml file)
wuzzy_hydra_map = {
    "fuzz.WRatio": fuzz.WRatio,
    "fuzz.partial_ratio": fuzz.partial_ratio,
}

## EvaluationData: inTESTED #####################################################################
class EvaluationData:
  def __init__(self, n_matches=0, n_targets=0, n_predictions=0):      
      self.n_matches = n_matches
      self.n_targets = n_targets
      self.n_predictions = n_predictions

  def add_evaluation(self, other):
      """Adds other EvaluationData."""
      self.n_matches += other.n_matches
      self.n_targets += other.n_targets
      self.n_predictions += other.n_predictions

  def to_tuple(self):
      """Returns the evaluation as a tuple (for e.g. JSON serialization)."""
      return (self.n_matches, self.n_targets, self.n_predictions)

  def __repr__(self):
      """Returns a string representation of the evaluation."""
      return f"{{'n_matches': {self.n_matches}, 'n_targets': {self.n_targets}, 'n_predictions': {self.n_predictions}}}"

  def all_rejected(self):
      return self.n_matches == 0 and self.n_targets > 0


## calculate_recall: xxx #####################################################################
def calculate_recall(evaldata: EvaluationData):
    # Recall: matches / targets
    tp = evaldata.n_matches
    fn = evaldata.n_targets - evaldata.n_matches
    if tp == 0:
        return 0
    return tp / (tp + fn)

## calculate_precision: xxx #####################################################################
def calculate_precision(evaldata: EvaluationData):
    # Precision: matches / predictions
    tp = evaldata.n_matches
    fp = evaldata.n_predictions - evaldata.n_matches

    if tp == 0:
        return 0
    return tp / (tp + fp)

## calculate_f1_score: xxx #####################################################################
def calculate_f1_score(evaldata: EvaluationData):
    precision = calculate_precision(evaldata)
    recall = calculate_recall(evaldata)

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

## generate_metrics_report: xxx #####################################################################
def generate_metrics_report(evaluation) -> str:
    """Generates a string with the evaluation metrics for each category."""
    report_str = ""
    for category, values in evaluation.items():
        tp, ap, pp = values  # Unpack the true positives, actual positives, and predicted positives
        recall = calculate_recall(tp, ap)
        precision = calculate_precision(tp, pp)
        f1_score = calculate_f1_score(precision, recall)
        
        report_str += f"Category: {category}\n"
        report_str += f"Recall: {recall:.2f}, Precision: {precision:.2f}, F1 Score: {f1_score:.2f}\n\n"
            
    return report_str

## deep_compare_compendiums: xxx #####################################################################
def deep_compare_compendiums(obj1, obj2, path=""):
    """Recursively compare two objects to check if they are identical, printing all differences."""
    differences = []

    if isinstance(obj1, (int, float, str)) and isinstance(obj2, (int, float, str)):
        try:
            if int(obj1) != int(obj2):
                differences.append(f"    Num val. mismatch -> {path}: {obj1} != {obj2}")
            return differences
        except ValueError:
            pass  # If conversion fails, fall back to standard comparison

    if type(obj1) != type(obj2):
        differences.append(f"    Type mismatch -> {path}: {type(obj1).__name__} != {type(obj2).__name__}")
        return differences

    if isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            differences.append(f"    Key mismatch -> {path}: {obj1.keys()} != {obj2.keys()}")
        for key in obj1:
            child_differences = deep_compare_compendiums(obj1[key], obj2[key], path + f"['{key}']")
            differences.extend(child_differences)
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            differences.append(f"    List size mismatch -> {path}: {len(obj1)} != {len(obj2)}")
        for index, (item1, item2) in enumerate(zip(obj1, obj2)):
            child_differences = deep_compare_compendiums(item1, item2, path + f"[{index}]")
            differences.extend(child_differences)
    # isinstance(obj1, EvaluationData) is False because metrics.py has been previously imported by data_utils.py
    # so obj1 is of type: <class 'src.data.metrics.EvaluationData'> and not <class 'EvaluationData'>
    # A (not pretty yet pragmatic) workaround is to check if the object has the method <to_tuple>
    elif callable(getattr(obj1, 'to_tuple', None)):
        if obj1.to_tuple() != obj2.to_tuple():
            differences.append(f"    EvaluationData() mismatch -> {path}: {obj1.to_tuple()} != {obj2.to_tuple()}")
    else: # Fallback: should not be reached 
        if obj1 != obj2:
            differences.append(f"    Value mismatch -> {path}: {obj1} != {obj2}")

    return differences

## compare_compendiums: xxx #####################################################################
def compare_compendiums(comp_A_file, comp_B_file):
    """Compares two compendium structures, printing all differences."""

    try:
      with open(comp_A_file, 'rb') as file:
        comp_A_data = pickle.load(file)
    except FileNotFoundError:
      logger.error(f"File not found: {comp_A_file}")
      return "Error catched."
    except Exception as e:
      logger.error(f"Error loading file: {e}")
      return "Error catched."

    try:
      with open(comp_B_file, 'rb') as file:
        comp_B_data = pickle.load(file)
    except FileNotFoundError:
      logger.error(f"File not found: {comp_B_file}")
      return "Error catched."
    except Exception as e:
      logger.error(f"Error loading file: {e}")
      return "Error catched."

    differences = []

    # Compare overall evaluation
    try:
      overall_differences = deep_compare_compendiums(comp_A_data['eval'], comp_B_data['eval'], "['eval']")
      if overall_differences:
          differences.append("\nOverall differences:\n====================")
      differences.extend(overall_differences)
    except Exception as e:
      logger.error(f"Error comparing compendiums: {e}")
      return "Error catched."

    # Compare all articles and quotes
    try:
      art_dif_displayed = False
      for index, (articleA, articleB) in enumerate(zip(comp_A_data['art'], comp_B_data['art'])):
          current_path = f"['art'][{index}]"
          article_differences = deep_compare_compendiums(articleA, articleB, current_path)
          if article_differences:
              if not art_dif_displayed:
                differences.append("\nArticle differences:\n====================")
                art_dif_displayed = True
              # Display the article index if any differences are found
              differences.append(f"Article[{comp_A_data['art'][index]['DRidx']}]:")
          differences.extend(article_differences)

    except Exception as e:
      logger.error(f"Error comparing compendiums: {e}")
      return "Error catched."

    if differences:
        differences_str = '\n'.join(differences)
        return differences_str
    return None


## generate_compendium_report: xxx #####################################################################
def generate_compendium_report(data_dir, compendium_eval, DR_indices, missing_p_indices):
    
    report_lines = ["COMPENDIUM:  | " + " ".join(f"{cat:5}" for cat in ARTICLE_CATEGORIES)]
    report_lines.append("-" * 62)

    def generate_metrics_lines(eval_data, categories, include_metrics=True, isquote=False):
        lines = []
        if isquote:
           spacing = 20
        else:
           spacing = 12
        # Ensure eval_data for ARTICLE_CATEGORIES, provide defaults if missing
        eval_data_full = {cat: eval_data.get(cat, EvaluationData()) for cat in categories}

        for n_counts in ["Matches:", "Targets:", "Predictions:"]:
            attr_name = 'n_' + n_counts[:-1].lower()
            values = [f"{getattr(eval_data_full[cat], attr_name):<5}" for cat in categories]
            lines.append(f"{n_counts:<{spacing}} | " + " ".join(values))

        if include_metrics:
          for metric_func, metric_name in zip([calculate_precision, calculate_recall, calculate_f1_score], ["Precision:", "Recall:", "F1 Score:"]):
              values = [f"{metric_func(eval_data_full[cat]):<5.2f}" for cat in categories]
              lines.append(f"{metric_name:<{spacing}} | " + " ".join(values))
        return lines

    report_lines.extend(generate_metrics_lines(compendium_eval['eval'], ARTICLE_CATEGORIES, include_metrics=True))

    report_lines.append(f"\nTotal articles to predict: {len(missing_p_indices) + len(DR_indices)}")
    report_lines.append(f"Missing predicted articles: {len(missing_p_indices)}")
    if missing_p_indices:
      report_lines.append(f'{missing_p_indices}')

    for article_eval in compendium_eval['art']:
        art_idx = article_eval['DRidx']
        report_lines.append(f"\nARTICLE[{art_idx:03d}] | " + " ".join(f"{cat:5}" for cat in ARTICLE_CATEGORIES))
        report_lines.append("-" * 62)
        report_lines.extend(generate_metrics_lines(article_eval, ARTICLE_CATEGORIES, include_metrics=False))

        if 'QUOTES' in article_eval and article_eval['QUOTES']:
            for quote_eval in article_eval['QUOTES']:                
                quote_target_idx = quote_eval['DRidx']['t']
                quote_predicted_idx = quote_eval['DRidx']['p']
                report_lines.append(f"\n    QUOTE[t:{quote_target_idx:02d} -> p:{quote_predicted_idx:02d}]  | " + " ".join(f"{cat:5}" for cat in QUOTE_CATEGORIES))
                report_lines.append("    " + "-" * 51)
                report_lines.extend(["    " + line for line in generate_metrics_lines(quote_eval, QUOTE_CATEGORIES, include_metrics=False, isquote=True)])


    compendium_report = '\n'.join(report_lines)

    report_filename = "compendium_report.txt"
    with open(os.path.join(data_dir, report_filename), 'w') as file:
        file.write(compendium_report)

    return compendium_report

## make_article_evaluation_serializable: xxxx #####################################################################
def make_article_evaluation_serializable(obj):
    """Converts an EvaluationData object to a tuple, and recursively converts all nested objects."""
    # Future proof for nested objects.

    if isinstance(obj, EvaluationData):
        return obj.to_tuple()  # Convert directly if it's an instance
    elif isinstance(obj, dict):
        # Convert each value in the dictionary, recursively
        return {k: make_article_evaluation_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Convert each item in the list, recursively
        return [make_article_evaluation_serializable(v) for v in obj]
    else:
        return obj

## relaxed_entity_matching: inTESTED #####################################################################
def relaxed_entity_matching(target_list, predicted_list, threshold, key):

    if key == "GEN":
      # If gender is "X", it is considered a match
      if  target_list[0] == predicted_list[0] or\
         (target_list[0] == "X" and predicted_list[0]) or\
         (target_list[0] and predicted_list[0] == "X"):
        return 1
      else:
        return 0

    elif key == "FUN":
      # If both target and predicted are in the same relaxed group, it is considered a match
      relaxed_groups = [['Politician', 'Other'],
                        ['Expert', 'Professional expert', 'Other'],
                        ['Case', 'DR source', 'Other'],
                        ['Authority', 'Other'],
                        ['Interest organization', 'Other']]
      for group in relaxed_groups:
        if target_list[0] in group and predicted_list[0] in group:
          return 1
      return 0
        
    else: # [PER, ORG, SRC, ROL, EMP]
      potential_matches = []     

      # First pass: Identify potential matches with their scores
      for pred in predicted_list:
          # If prediction is "", then skip FuzzyWuzzy extractOne() to avoid warning:
          # "Applied processor reduces input query to empty string, 
          # all comparisons will have score 0. [Query: '']
          # Empty prediction should actually be [] rather than "" or [""].
          if pred != "": 
            match = process.extractOne(pred, target_list, score_cutoff=threshold)
          else:
            match = None
          if match:
              potential_matches.append((pred, match))
      
      # Sort potential matches by score in descending order
      potential_matches.sort(key=lambda x: x[1][1], reverse=True)
    
      refined_matches = 0
      used_targets = []

      # Process matches
      for pred, (best_match, score) in potential_matches:
          if best_match not in used_targets:
              refined_matches += 1
              used_targets.append(best_match)

    return refined_matches

## evaluate_entity_list: TESTED #####################################################################
def evaluate_entity_list(target_list, predicted_list, wuzzy_conf, key):
  """Evaluate a list of entities e.g.["Peter", "Jens Jensen"] using strict or relaxed matching."""
  logger = logging.getLogger(logging_config.logger_name)


  # Ensure target_list and predicted_list are lists (to deal with GEN and FUN)
  if not isinstance(target_list, list):
      target_list = [target_list]
  if not isinstance(predicted_list, list):
      predicted_list = [predicted_list]


  wuzzy_mode = wuzzy_conf['mode']
  threshold = wuzzy_conf.get('threshold', 0) # Default to 0 if not specified (for GEN+FUN strict mode)

  n_matches = 0

  if wuzzy_mode == 'strict':
      # In strict mode, directly count matches
    n_matches = sum(pred in target_list for pred in predicted_list)
  elif wuzzy_mode == 'relaxed':
    n_matches = relaxed_entity_matching(target_list, predicted_list, threshold, key)
  else:
    logger.error(f"Unknown wuzzy mode: {wuzzy_mode} (must be 'strict' or 'relaxed').")
    n_matches = 0

  n_predictions = len(predicted_list)
  n_targets = len(target_list)

  return EvaluationData(n_matches, n_targets, n_predictions)

## calculate_quote_similarity_matrix: TESTED ########################################################
def calculate_quote_similarity_matrix(target_article, predicted_article, ratio_fcn):
  """Calculate similarity matrix for quotes to then find matching quotes."""

  similarity_matrix = np.zeros((len(target_article['QUOTES']), len(predicted_article['QUOTES'])), dtype=int)
  
  for t_idx, target_quote in enumerate(target_article['QUOTES']):
      for p_idx, predicted_quote in enumerate(predicted_article['QUOTES']): 
          
          # ratio_fcn like Wratio and partial_ratio, consider a substring as 100% match
          # so two predicted quotes, one being a substring of the target quote, would be 100% match
          # and the other predicted quote (which is 100% correct) would not be considered at all.
          # therefore this logic is needed to avoid 100% match for a substring 
          if target_quote['TEXT'] == predicted_quote['TEXT']:
            score = 100
          else:
            # cap to 99 to avoid 100% match for a substring
            score = min(ratio_fcn(target_quote['TEXT'], predicted_quote['TEXT']), 99) 

          similarity_matrix[t_idx][p_idx] = score
  
  return similarity_matrix


## find_matching_quotes: TESTED #####################################################################
def find_matching_quotes(target_article, predicted_article, wuzzy_conf):
    logger = logging.getLogger(logging_config.logger_name)

    threshold = wuzzy_conf['TEXT']['threshold']
    ratio_fcn = wuzzy_hydra_map[wuzzy_conf['TEXT']['ratio_fcn']]

    similarity_matrix = calculate_quote_similarity_matrix(target_article, predicted_article, ratio_fcn)
    logger.debug(f"similiarity matrix:\n{similarity_matrix}")

    # Step 1: Collect all pairs with their scores, where scores are above the threshold
    potential_pairs = []
    for i in range(similarity_matrix.shape[0]):      # For each target quote
        for j in range(similarity_matrix.shape[1]):  # For each predicted quote
            score = similarity_matrix[i, j]
            if score >= threshold:
                potential_pairs.append(((i, j), score))
    
    # Step 2: Sort the potential pairs by score in descending order
    potential_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Inform whether the threshold is too low:
    tmp_matrix = similarity_matrix.copy()
    np.fill_diagonal(tmp_matrix, 0)
    # Extract off-diagonal scores
    off_diag_scores = tmp_matrix[tmp_matrix != 0]

    smallest_diag_score = np.min(np.diag(similarity_matrix))
    if np.all(off_diag_scores < smallest_diag_score):
      if smallest_diag_score > threshold:
        logger.notice(f"Quote threshold could be increased: {threshold} -> {smallest_diag_score}")
      else:
        logger.notice(f"Quote threshold: {threshold} is TIGHT")


    # TODO: The intention is good but the logic is flawed.
    # I thought the diagonal of the similarity matrix would contain the scores of the best matches.
    # Code commented out for now.
    # n_quotes_below_threshold = sum(np.diag(similarity_matrix) < threshold)
    # if n_quotes_below_threshold:    
    #     logger.warning(f"{n_quotes_below_threshold} quote(s) below {threshold} were excluded: ({np.diag(similarity_matrix)}).")  
    #     print("TESTING")
    #     print(f"{n_quotes_below_threshold} quote(s) below {threshold} were excluded: ({np.diag(similarity_matrix)}).")
    
    # Step 3: Select pairs ensuring no target or predicted quote is used more than once
    # TODO: Consider a better logic if e.g. two predicted quotes are very similar to the same target quote
    # risking to exclude the second predicted quote, instead of matching it to a different target quote.
    selected_pairs = []
    used_targets = set()
    used_predicted = set()
    for (i, j), score in potential_pairs:
        if i not in used_targets and j not in used_predicted:
            selected_pairs.append(((i, j), score))
            used_targets.add(i)
            used_predicted.add(j)

    logger.debug(f"Target quotes:    {len(target_article['QUOTES']):2d}\n"\
                 f"Predicted quotes: {len(predicted_article['QUOTES']):2d}\n"\
                 f"Matched quotes:   {len(selected_pairs):2d}")

    if len(selected_pairs) < len(potential_pairs):
      logger.notice(f"{len(selected_pairs)} quote(s) were selected out of {len(potential_pairs)} potential pairs.\n"\
                     "There might be multiple predicted quotes matching the same target quote (or vice versa).")        
            
    return selected_pairs


## evaluate_quotes: TESTED #####################################################################
def evaluate_quotes(target_article, predicted_article, wuzzy_conf):
    logger = logging.getLogger(logging_config.logger_name)

    text_evaluation = EvaluationData()
    quote_evaluations = []
    DR_indices = []
    matching_quotes = find_matching_quotes(target_article, predicted_article, wuzzy_conf)

    # Update text_evaluation with matching quotes (n_matches, n_targets, n_predictions)
    text_evaluation.add_evaluation(EvaluationData(len(matching_quotes), len(target_article['QUOTES']), len(predicted_article['QUOTES'])))


    # Evaluate and accumulate evaluations for each category in matching quotes
    for (t_idx, p_idx), score in matching_quotes:
        logger.debug(f"==== QUOTE (score: {int(score):3d}) ====")
        logger.debug(f"Target:{target_article['QUOTES'][t_idx]}")
        logger.debug(f"Predicted:{predicted_article['QUOTES'][p_idx]}")

        single_quote_evaluation = {category: EvaluationData() for category in QUOTE_CATEGORIES}

        missing_key = False
        for quote_key in QUOTE_CATEGORIES:
            if quote_key in target_article['QUOTES'][t_idx] and quote_key in predicted_article['QUOTES'][p_idx]:
                entity_evaluation = evaluate_entity_list(target_article['QUOTES'][t_idx][quote_key],\
                                                         predicted_article['QUOTES'][p_idx][quote_key],\
                                                         wuzzy_conf[quote_key],quote_key)

                single_quote_evaluation[quote_key].add_evaluation(entity_evaluation)
                logger.debug(f"Quote[{quote_key}]: {entity_evaluation}")
            else:
               logger.notice(f"{quote_key} not found in quote")
               missing_key = True
        if missing_key:
          logger.notice(f"Missing keys in quote: '{target_article['QUOTES'][t_idx]['TEXT']}'") 

        DR_indices.append((t_idx, p_idx))
        quote_evaluations.append(single_quote_evaluation)

    return text_evaluation, quote_evaluations, DR_indices


## evaluate_article: TESTED #####################################################################
def evaluate_article(target_article, predicted_article, wuzzy_conf):
  logger = logging.getLogger(logging_config.logger_name)

  article_evaluation = {**{category: EvaluationData() for category in ARTICLE_CATEGORIES}, "QUOTES": []}

  # Evaluate PER and add to article_evaluation
  art_key = "PER"
  PER_evaluation = evaluate_entity_list(target_list=target_article[art_key], predicted_list=predicted_article[art_key],
                                        wuzzy_conf=wuzzy_conf[art_key], key=art_key)
  article_evaluation["PER"] = PER_evaluation
  logger.debug(f"PER_evaluation: {PER_evaluation}")

  # Evaluate ORG and add to article_evaluation
  art_key = "ORG"
  ORG_evaluation = evaluate_entity_list(target_list=target_article[art_key], predicted_list=predicted_article[art_key],
                                        wuzzy_conf=wuzzy_conf[art_key], key=art_key)
  article_evaluation["ORG"] = ORG_evaluation
  logger.debug(f"ORG_evaluation: {ORG_evaluation}")


  if target_article.get('QUOTES') and predicted_article.get('QUOTES'):
    text_evaluation, quote_evaluations, DR_indices = evaluate_quotes(target_article, predicted_article, wuzzy_conf)

    article_evaluation["TEXT"].add_evaluation(text_evaluation)
    # Add quote evaluations separately to the article so they can be accessed like this: 
    # ['QUOTES'][idx] and ['QUOTES'][idx]['GEN...EMP']
    for single_quote_eval, DR_idx in zip(quote_evaluations, DR_indices):
      single_quote_eval['DRidx'] = {'t': DR_idx[0], 'p': DR_idx[1]}
      article_evaluation["QUOTES"].append(single_quote_eval)

    logger.debug(f"TEXT evaluation: {text_evaluation}")
    logger.debug(f"QUOTE evaluations: {quote_evaluations}")

    # Add quote evaluations to article_evaluation
    for quote_category in QUOTE_CATEGORIES:
        category_evaluation = EvaluationData() 
        for quote_eval in article_evaluation["QUOTES"]:
            category_evaluation.add_evaluation(quote_eval[quote_category])
        article_evaluation[quote_category].add_evaluation(category_evaluation)

  else:
    quotes_not_evaluated = EvaluationData(0, len(target_article['QUOTES']), len(predicted_article['QUOTES']))
    article_evaluation["TEXT"].add_evaluation(quotes_not_evaluated)

  return article_evaluation

## get_DR_indices_from_folder: xxxx #####################################################################
def get_DR_indices_from_folder(data_dir, file_forename):
  logger = logging.getLogger(logging_config.logger_name)
  
  # Find target and predicted files 
  file_pattern = re.compile(rf"{file_forename}(\d+)[pt]\.json")

  # Track occurrences of DR indices with 't'arget and 'p'redicted
  occurrences = {}

  try:
    for filename in os.listdir(data_dir):
        match = file_pattern.match(filename)
        if match:
          index = int(match.group(1))
          file_type = filename[-6]  # Fetch 'p' or 't'          
          if index not in occurrences:
              occurrences[index] = set()
          occurrences[index].add(file_type)
  except Exception as e:
    logger.error(f"Error finding DR indices: {e}")

  # Filter indices with both 'p' and 't'
  indices = {index for index, types in occurrences.items() if 'p' in types and 't' in types}
  sorted_indices = sorted(indices)

  # Find and count missing 'p' files/indices 
  missing_p_indices = [index for index, types in occurrences.items() if 'p' not in types and 't' in types]
  missing_p_indices = sorted(missing_p_indices)
  if missing_p_indices:
    for missing_index in missing_p_indices:
      logger.notice(f'Missing {file_forename}{missing_index}p.json')


  return sorted_indices, missing_p_indices

## evaluate_compendium_from_folder: xxxx #####################################################################
def evaluate_compendium_from_folder(cfg: DictConfig):
  data_dir = hh.get_data_dir(cfg)
  print(f"Data dir: {data_dir}")

  wuzzy_conf = cfg.data.wuzzy_conf

  logger = logging.getLogger(logging_config.logger_name)

  file_forename = "article_"    
  base_filepath = os.path.join(data_dir, f"{file_forename}")
  DR_indices, missing_p_indices = get_DR_indices_from_folder(data_dir, file_forename)

  # Compendium stores the evaluation of all articles in 'art' and the overall evaluation of the compendium in 'eval'
  # Each 'art' contains the corresponding overall evaluation of the article. 
  # For each 'art', 'QUOTES' contains the evaluation of each quote.
  compendium_eval = {'art': [], 'eval' : {category: EvaluationData() for category in ARTICLE_CATEGORIES}}
  
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
          continue        

        logger.notice(f"\nEvaluating art.[{DR_idx}]: {target_article['headline'][:60]}...")

        # Load predicted article
        predicted_article = du.load_article_from_json(base_filepath_idx + "p.json")
        if predicted_article is None:
          continue        

        # Evaluate article
        article_evaluation = evaluate_article(target_article, predicted_article, wuzzy_conf)
        # Add true DR-data index to article_evaluation
        article_evaluation['DRidx'] = DR_idx          
        compendium_eval['art'].append(article_evaluation)

        # Update compendium with article evaluation
        for category, eval_data_dict in compendium_eval['art'][-1].items():
            if category != "QUOTES" and category in compendium_eval['eval']:
                eval_data = EvaluationData(eval_data_dict.n_matches, eval_data_dict.n_targets, eval_data_dict.n_predictions)
                compendium_eval['eval'][category].add_evaluation(eval_data)

      except Exception as e:
          logger.error(f"Error evaluating article[{DR_idx}]: {e}")
          continue
      
    # Save compendium and generate report
    pickle_filename = os.path.join(data_dir, "compendium_eval.pkl") 

    with open(pickle_filename, 'wb') as file:
      pickle.dump(compendium_eval, file)
      logger.notice("Compendium evaluation saved.")

    generate_compendium_report(data_dir, compendium_eval, DR_indices, missing_p_indices)

  return compendium_eval


## run_main: xxxx  #####################################################################
#@hh.can_run_as_standalone(config_path="../../conf")
def main(cfg: DictConfig = None):
  """main function: cfg passed from main_openai.py."""

  data_dir = hh.get_data_dir(cfg)

  comp_file_openai = os.path.join(data_dir, "compendium_openai.pkl")
  comp_file_other = os.path.join(data_dir, "compendium_eval.pkl")

  print(f"Comparing compendiums:\n{comp_file_openai}\n{comp_file_other}")

  dif_str = compare_compendiums(comp_file_openai, comp_file_other)

  print(f"Difference report: {dif_str}")


if __name__ == "__main__":
  main()