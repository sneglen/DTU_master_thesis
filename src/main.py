# standard libraries
import os

# third-party libraries
import hydra
from omegaconf import DictConfig

# local libraries
from src.config.logging_config import initiate_logging
import src.openai.openai_integration as openai_int
import src.data.metrics as mt
import src.data.data_utils as du  
import src.data.report_figures as rf  
import src.data.dpo_utils as dpo
import src.utils.hydra_helper as hh
import src.spe.spe_manager as spe_manager
from src.utils.misc import is_VM

if is_VM():
  import src.finetune.llm_sft_trainer as llm_sft_trainer
  import src.finetune.llm_dpo_trainer as llm_dpo_trainer

  import src.finetune.finetuned_inference as test_finetuned_inference

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
  logger = initiate_logging(cfg)
  
  data_dir = hh.get_data_dir(cfg)
  logger.notice(f"Data dir: {data_dir}")

## TASK: openai_annotate  @ PC
  if cfg.main_task.openai_annotate == 'run':
    logger.notice("TASK: openai_annotate")
    openai_int.annotate_articles(cfg)

## TASK: evaluate_compendium @ PC
  if cfg.main_task.evaluate_compendium == 'run':
    logger.notice("TASK: evaluate_compendium")
    mt.evaluate_compendium_from_folder(cfg)

## TASK: generate_rejected_dpo_data @ PC
  if cfg.main_task.generate_rejected_dpo_data == 'run':
    logger.notice("TASK: generate_rejected_dpo_data")
    dpo.generate_rejected_dpo_data(cfg)

## TASK: compare_compendiums @ PC
  if cfg.main_task.compare_compendiums == 'run':
    logger.notice("TASK: compare_compendiums")
    # keep this code as safety check (to be removed in future)
    comp_file_openai = os.path.join(data_dir, "compendium_openai.pkl")
    comp_file_other = os.path.join(data_dir, "compendium_eval.pkl")

    print(f"Comparing compendiums:\n{comp_file_openai}\n{comp_file_other}")
    dif_str = mt.compare_compendiums(comp_file_openai, comp_file_other)
    print(f"Difference report: {dif_str}")
  
## TASK: data_stat_analysis @ PC
  if cfg.main_task.data_stat_analysis == 'run':
    du.main(cfg=cfg)

## TASK: report_figures @ PC
  if cfg.main_task.report_figures == 'run':
    rf.main(cfg=cfg)

## TASK: local LLM @ Google VM
  if cfg.main_task.spe_tasks == 'run':
    spe_manager.run_tasks(cfg=cfg)

## TASK: Fine-tune LLM @ Google VM
  if cfg.main_task.fine_tune_llm_sft == 'run':
    llm_sft_trainer.train_llm(cfg=cfg)

## TASK: Fine-tune LLM @ Google VM
  if cfg.main_task.fine_tune_llm_dpo == 'run':
    llm_dpo_trainer.train_llm(cfg=cfg)

## TASK: TESTING PURPOSES @ Google VM
  if cfg.main_task.test_finetuned_inference == 'run':
    test_finetuned_inference.main(cfg=cfg)

  logger.notice("\n<SESSION ENDED>")

if __name__ == "__main__":
  main()

