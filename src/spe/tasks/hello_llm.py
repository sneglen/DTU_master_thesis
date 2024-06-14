import sglang as sgl
from omegaconf import DictConfig
import logging


logger = logging.getLogger(__name__)

@sgl.function
def worker_text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n")

    
def hello_llm(cfg: DictConfig):
  try:
    logger.notice("BEGIN task: hello_llm...")

    state = worker_text_qa.run(
        question="Hvor ligger København?",
        temperature=0.01,
        stream=False
    )

    for out in state.text_iter():
        print(out, end="", flush=True)
    print("\n")

  except Exception as e:
    logger.error(f"Error: {e}")
  
  finally:
    logger.notice("END task: hello_llm.")