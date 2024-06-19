## https://platform.openai.com/docs/api-reference/making-requests

# standard libraries
import json
import sys
import logging
import os
import time

# third-party libraries
from openai import OpenAI
from omegaconf import DictConfig
import pickle

# local libraries
import src.data.data_utils as du
import src.data.metrics as mt
import src.utils.hydra_helper as hh 
from src.config import logging_config
from src.data.metrics import EvaluationData
from src.data.articles_dataset import ARTICLE_CATEGORIES


# path to "type_definitions/", where QuoteCollection is defined, needed to load pickle files.
sys.path.append('src/data/dr_lib/')

logger = logging.getLogger(logging_config.logger_name)

def load_system_content(cfg: DictConfig) -> str:
    """Load system content from file."""

    system_content_file = f"{cfg.openai.query_dir}{cfg.openai.query_name}_v{cfg.openai.query_ver}.txt"

    try:
        with open(system_content_file, 'r') as file:
            system_content = file.read()
    except FileNotFoundError:
          raise FileNotFoundError(f"File not found: {system_content_file}")
    except Exception as e:
          raise Exception(f"An error occurred while reading the file: {e}")
                            
    return system_content


def send_openai_request(cfg, client, system_content, user_content): 
 
  model_tag = cfg.llm.tag.gpt
  model = cfg.llm[model_tag].name

  response = client.chat.completions.create(
    model=model,
    response_format={"type":"json_object"},
    messages=[
      {"role": "system", "content": system_content},
      {"role": "user", "content": user_content}
    ]
  )

  prompt_tokens = response.usage.prompt_tokens
  completion_tokens = response.usage.completion_tokens
  consumed_tokens = prompt_tokens + completion_tokens
  logger.info(f"[Prompt | Completion | Total]: [{prompt_tokens} | {completion_tokens} | {consumed_tokens}] tokens")
    
  return response, consumed_tokens

## articles LOOP: annotate, evaluate, save
def annotate_articles(cfg: DictConfig):
    data_dir = hh.get_data_dir(cfg)

    client = OpenAI()
    system_content = load_system_content(cfg)

    # Get datasplit to annotate (articles and indices)
    selected_split = cfg.openai.annotate.split  
    articles = du.load_articles(cfg.data.split[selected_split])
    DR_indices = hh.parse_article_DR_indices(cfg.openai.annotate.indices)


    target_article_dict = predicted_article_dict = article_evaluation = "offline run"    
    file_forename = "article_"    
    base_filepath = os.path.join(data_dir, f"{file_forename}")

    total_tokens = 0
    consumed_tokens = 0
    missing_p_indices = []

    compendium_eval = {'art': [], 'eval' : {category: EvaluationData() for category in ARTICLE_CATEGORIES}}

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
          user_content = target_article_dict['TEXT']
  
          # SAVE target article
          du.save_article_to_json(base_filepath_idx + "t.json", target_article_dict)

          current_time = time.time()
          elapsed_time = current_time - start_time
          start_time = time.time()
          total_elapsed_time += elapsed_time

          avg_time = total_elapsed_time / processed_articles if processed_articles > 0 else 0
          msg = f"\nProcessing art.[{DR_idx}], dt: {elapsed_time:.0f}[s]. avg: {avg_time:.0f}[s]: '{target_article_dict['headline'][:60]}...'"
          logger.notice(msg)
          logger.notice("=" * (len(msg)-1))

          if cfg.openai.query_online:
            # SEND LLM request
            raw_response, consumed_tokens = send_openai_request(cfg, client, system_content, user_content)
            total_tokens += consumed_tokens

            # SAVE LLM's raw response
            with open(base_filepath_idx + "r.pkl", 'wb') as file:
              pickle.dump(raw_response, file)
          else:
            # LOAD LLM's raw response
            with open(base_filepath_idx + "r.pkl", 'rb') as file:
              raw_response = pickle.load(file)

          json_str_response = raw_response.choices[0].message.content

          # SAVE LLM's structured response
          du.save_json_str_to_file(base_filepath_idx + "s.json", json_str_response)

          # CLEAR target data: PER, ORG and QUOTES
          predicted_article_dict, _ = du.prepare_article_for_prediction(target_article_dict)

          # SAVE cleared article (to have certainty that predicted article is not an accidental copy of target article)
          du.save_article_to_json(base_filepath_idx + "c.json", predicted_article_dict)

          # UPDATE predicted article with OpenAI's response
          predicted_entities_dict = json.loads(json_str_response)
          predicted_article_dict = du.update_with_predicted_entities(predicted_entities_dict, predicted_article_dict)

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
    logger.notice(msg)
    logger.notice("=" * (len(msg)-1))

    pickle_filename = os.path.join(data_dir, "compendium_openai.pkl") 

    with open(pickle_filename, 'wb') as file:
      pickle.dump(compendium_eval, file)
    logger.notice("Compendium evaluation saved.")

    mt.generate_compendium_report(data_dir, compendium_eval, DR_indices, missing_p_indices)


def run_integration(cfg: DictConfig, data_dir: str):

  annotate_articles(cfg=cfg)
  mt.evaluate_compendium_from_folder(cfg) 
  
  
if __name__ == "__main__":
  run_integration()
