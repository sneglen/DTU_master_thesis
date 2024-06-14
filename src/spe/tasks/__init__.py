from .download_llm import download_llm
from .local_tests import local_tests
from .hello_llm import hello_llm
from .multiturn_questions import multiturn_questions
from .generate_character import generate_character
from .annotate_articles import annotate_articles


class task:
  download_llm = download_llm
  local_tests = local_tests
  hello_llm = hello_llm
  multiturn_questions = multiturn_questions
  generate_character = generate_character
  annotate_articles = annotate_articles