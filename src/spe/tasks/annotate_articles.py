# standard libraries
import sys
import os
import json
import logging
import time

# third-party libraries
from omegaconf import DictConfig
import pickle

# local libraries
import src.data.data_utils as du
import src.utils.hydra_helper as hh
import src.data.metrics as mt
from src.data.metrics import EvaluationData
from src.data.articles_dataset import ARTICLE_CATEGORIES
from src.spe.annotator_agent_manager import annotator_agent_manager

# path to "type_definitions/", where QuoteCollection is defined, needed to load pickle files.
sys.path.append('src/data/dr_lib/')


logger = logging.getLogger(__name__)

## articles LOOP: annotate, evaluate, save
def annotate_articles(cfg: DictConfig):
    data_dir = hh.get_data_dir(cfg)

    # Get datasplit to annotate (articles and indices)
    selected_split = cfg.spe.annotate.split  
    articles = du.load_articles(cfg.data.split[selected_split])
    DR_indices = hh.parse_article_DR_indices(cfg.spe.annotate.indices)

    target_article_dict = predicted_article_dict = article_evaluation = "offline run"
    
    file_forename = "article_"    
    base_filepath = os.path.join(data_dir, f"{file_forename}")

    total_tokens = 0

    compendium_eval = {'art': [], 'eval' : {category: EvaluationData() for category in ARTICLE_CATEGORIES}}

    missing_p_indices = []

    if not DR_indices:
      logger.notice(f"No articles found to process: <{data_dir}>.")
      return
    else:
      start_time = time.time()
      processed_articles = 0
      total_elapsed_time = 0
      for DR_idx in DR_indices:
        try:
          base_filepath_idx = base_filepath + str(DR_idx)

          # Load target article (and user content)
          target_article_dict = articles[DR_idx]  
  
          # SAVE target article
          du.save_article_to_json(base_filepath_idx + "t.json", target_article_dict)

          current_time = time.time()
          elapsed_time = current_time - start_time
          start_time = time.time()
          total_elapsed_time += elapsed_time

          avg_time = total_elapsed_time / processed_articles if processed_articles > 0 else 0
          msg = f"Processing art.[{DR_idx}], dt: {elapsed_time:.0f}[s]. avg: {avg_time:.0f}[s]: '{target_article_dict['headline'][:60]}...'"
          logger.notice(msg + "\n" + "=" * 78)

          # Retrieve response from LLM (online query if possible)
          try:
            json_dict_response, raw_response = annotator_agent_manager(cfg, target_article_dict)
            # save raw response to file (for debugging purposes and insight to create better dpo rejected entries)
            try:
              with open(base_filepath_idx + "raw.txt", "w") as file:
                file.write(raw_response)
            except Exception as e:
              logger.error(f"Could not save raw response to file: article [{DR_idx}]: {e}")
              continue

          except Exception as e:
            missing_p_indices.append(DR_idx)
            logger.error(f"Could not retrieve json_dict_response from response annotator_agent_manager: article [{DR_idx}]: {e}")
            logger.notice(f'It should not be possible to arrive here. json_dict_response: {json_dict_response}')
            continue  

          # SAVE LLM's structured response
          json_str_response = json.dumps(json_dict_response, indent=2, ensure_ascii=False)
          du.save_json_str_to_file(base_filepath_idx + "s.json", json_str_response)  

          # CLEAR target data: PER, ORG and QUOTES
          predicted_article_dict, _ = du.prepare_article_for_prediction(target_article_dict)

          # SAVE cleared article (to have certainty that predicted article is not an accidental copy of target article)
          du.save_article_to_json(base_filepath_idx + "c.json", predicted_article_dict)

          # UPDATE predicted article with annotator response
          predicted_article_dict = du.update_with_predicted_entities(json_dict_response, predicted_article_dict)

          # SAVE predicted article
          du.save_article_to_json(base_filepath_idx + "p.json", predicted_article_dict)

          article_evaluation = mt.evaluate_article(target_article_dict, predicted_article_dict, cfg.data.wuzzy_conf)
          article_evaluation['DRidx'] = DR_idx          
          compendium_eval['art'].append(article_evaluation)

          # Update compendium with article evaluation
          for category, eval_data_dict in compendium_eval['art'][-1].items():
              if category != "QUOTES" and category in compendium_eval['eval']:
                  eval_data = EvaluationData(eval_data_dict.n_matches, eval_data_dict.n_targets, eval_data_dict.n_predictions)
                  compendium_eval['eval'][category].add_evaluation(eval_data)

          # Save evaluation to JSON file
          du.save_article_evaluation_to_json(base_filepath_idx + "e.json", article_evaluation)

        except Exception as e:
          logger.error(f"Error processing article {DR_idx}: {e}")
          continue # continue with next article

        finally:
          processed_articles += 1

      logger.notice(f"Total tokens consumed: {total_tokens} (avg. {total_tokens/len(DR_indices):.0f}/article)")

    # Display total elapsed and avg. time for processing all articles
    total_elapsed_time += time.time() - start_time
    avg_time = total_elapsed_time / processed_articles if processed_articles > 0 else 0
    msg = f"\nFinished annotating. total dt: {total_elapsed_time:.0f}[s]. avg: {avg_time:.0f}[s]"
    logger.notice(msg + "\n" + "=" * 78)

    pickle_filename = os.path.join(data_dir, "compendium_spe.pkl") 

    with open(pickle_filename, 'wb') as file:
      pickle.dump(compendium_eval, file)
    logger.notice("Compendium evaluation saved.")

    mt.generate_compendium_report(data_dir, compendium_eval, DR_indices, missing_p_indices)
