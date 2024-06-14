import sglang as sgl
from omegaconf import DictConfig
import logging


logger = logging.getLogger(__name__)


@sgl.function
def worker_multiturn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))

def multiturn_questions(cfg: DictConfig):
  try:
    logger.notice("BEGIN task: multiturn_questions...")

    ## LLM Query:
    state = worker_multiturn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )   

    ## LLM Output:
    for m in state.messages():
        print(m["role"], ":", m["content"])

    print(state["answer_1"])

  except Exception as e:
    logger.error(f"Error: {e}")

  finally:
    logger.notice("END task: multiturn_questions.")
