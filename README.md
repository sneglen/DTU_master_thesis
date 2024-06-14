## Master Thesis Project at The Technical University of Denmark (DTU):

**Title:** 
AI-Driven Media Analysis: A case study of DR’s News Coverage

**Author:** Enrique Vidal Sanchez

**e-mail:** enrique@vidal.dk


## Project overview
The following figures provide an overview of the system's design and functionality, focusing on the architecture for annotating articles rather than delving into technical details. It highlights the three main software components of the project without going into technical details. The complete code is available in the repository, should the reader be interested in specific software parts. Note that the dataset is excluded from the repository, in line with DR's agreement.

First, it is shown how the articles are annotated using the LLMs from OpenAI. Following this, the architecture employing the open-source LLM is presented. Finally, an overview is given on how the open-source LLM is fine-tuned to enhance its annotation performance.

### 1) Annotation based on OpenAI's LLMs

![Alt text](docs/figures/openai_structure.png?raw=true "openai_structure")

The figure above illustrates the architecture designed to annotate the articles using OpenAI's models. Starting from the left, DR's dataset can be conceptually understood as divided in two parts: the articles to be annotated, and the (target) JSON structures containing the annotated information. Only the articles are processed by the $\texttt{Annotator Agent}$, which also fetches the annotation instructions. 

The annotator agent constructs a prompt for each article, incorporating the annotation instructions and the article itself. Each prompt is then sent to the LLM through OpenAI's API, which returns a response for each article.

If successful, the response contains a valid (predicted) JSON structure with the requested annotation information. 


After annotating the articles, the $\texttt{Evaluator}$ compares the target and predicted JSON structures and generates a report. Since the information to be evaluated not only involves multiclass classification but also extracting strings of varying lengths (such as names and quotes), heuristics are applied where the degree of exactitude can be specified. For instance, using the target quote $\textit{"Det var ikke en tur i parken..."}$, it would be reasonable for the evaluator to approve a predicted quote like $\textit{"- Det var ikke en tur i parken..."}$, which is preceded by a dash "-". However if the target gender is "Male" but is predicted as "Other", then it should be considered incorrect.

Thanks to OpenAI's API and  the capabilities of the GPT models, along with the prior conversion of the annotated information into suitable target JSON structures, the entire process becomes significantly more streamlined and reduces as a consequence the programming complexity.

### 2) Annotation based on $\texttt{DK-Mistral-7B}$

**Note**: For simplicity, Jurowetzki's LLM [[munin-neuralbeagle-7](https://huggingface.co/RJuro/munin-neuralbeagle-7b)], available on Hugging Face, is referred to as $\texttt{DK-Mistral-7B}$.

![Alt text](docs/figures/spe_llm_structure.png?raw=true "spe_llm_structure")


The overall purpose of the annotation task is the same as described in the previous section and hence they share common elements. The architecture begins similarly with DR's dataset, divided into articles and target JSON structure. 

The articles are retrieved by the $\texttt{Annotator Agent}$, but here the process differs. $\texttt{GPT-4}$ is estimated to have nearly 2 trillion parameters [[Wikipedia: GPT-4](https://en.wikipedia.org/wiki/GPT-4)], while $\texttt{DK-Mistral-7B}$ [[Mistral 7B - Jiang et al. 2023](https://arxiv.org/abs/2310.06825)], with its 7 billion parameters, is below 1\% of $\texttt{GPT-4}$'s size. Given the relatively small size of $\texttt{DK-Mistral-7B}$, it is more efficient to divide the annotation queries into smaller parts. Hence the relevant set of instructions needs to be fetched depending on the labels to be annotated. Moreover, the appropriate regex (regular expression) string also needs to be generated based on the actual label class structure. Recall that the LLM does not have inherent capabilities of returning JSON-formatted responses, and for this reason, the model needs to be $\textit{guided}$ based on the regex strings. 

The query is passed to $\texttt{SGLang}$'s API [[SGLang project](https://github.com/sgl-project/sglang)] and is sent to $\texttt{SGLang}$'s backend. The $\texttt{SGLang}$'s backend receives the query and passes it further to the LLM. 

Once the query is processed by the LLM, a response is received through $\texttt{SGLang}$'s API back into the $\texttt{Annotator agent}$. In there, the multiple JSON-formatted fragments are validated and then joined to form a complete valid (predicted) JSON structure with the requested annotation information for the article in question. 

Finally, after the annotation of the articles is concluded, the target and predicted JSON formatted data are evaluated in the same way as previously described in Section and a report is generated.

### 3) Fine-tuning $\texttt{DK-Mistral-7B}$

![Alt text](docs/figures/dpo_structure.png?raw=true "dpo_structure")

The third and final architecture component is related to $\texttt{DK-Mistral-7B}$'s fine-tuning. While the LLM is already versatile and capable of generating text, to enhance its performance on the specific annotation task, it is advantageous to fine-tune it for said purpose. Recall that for clarity the not fine-tuned model is referred to as $\texttt{base-DK-Mistral-7B}$ and the fine-tuned model is referred to as $\texttt{FT-DK-Mistral-7B}$.

There are several ways to fine-tune an LLM, one of which is  Reinforcement Learning from Human Feedback (RLHF), where an intelligent agent (the LLM to be fine-tuned in this context) is aligned to human preferences. However having a human in the loop undermines the purpose of the project which is precisely to develop an automated system that can perform annotation tasks without human intervention. 

A recent paradigm to fine-tune LLM from preferences without reinforcement learning is using Direct Policy Optimization(DPO), which solves the standard RLHF problem with only a simple classification loss [[Rafailov et al. - 2023](https://arxiv.org/abs/2305.18290)].

The optimization problem is converted to a binary classification task, where the model aims to maximize the likelihood of the chosen responses over the rejected ones. In practice DPO works by comparing two versions of the model's output and adjusting the model to prefer the chosen answer and to move away from the rejected answer. With virtually no tuning of hyperparameters, DPO is reported to perform similarly or better than existing RLHF algorithms. 

To construct a DPO dataset, the chosen (correct) answers are retrieved from the existing manually annotated DR dataset, see the figure above.

The rejected (wrong) answers are fetched from the outputs of $\texttt{base-Mistral-7B}$, as a result of the annotation process described in the previous section. Since not all the answers are wrong, to ensure a complete set of rejected answers, the missing rejected answers are automatically generated through string manipulation techniques. This automated process ensures that the project remains fully automated, aligning with the goal of eliminating manual intervention.

During training, the weights of $\texttt{FT-DK-Mistral-7B}$ are adjusted such that the likelihood of chosen responses is maximized, while the likelihood of the rejected responses is minimized. To prevent $\texttt{FT-DK-Mistral-7B}$ to "$\textit{forget}$" the originally learned language capabilities, $\texttt{base-DK-Mistral-7B}$ is included in the training loop as reference. 



## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   └── mkdocs.yml       <- Configuration file for mkdocs  
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── pyproject.toml       <- Project configuration file
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for MLOps.
