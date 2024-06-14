## This defines all targets as phony targets, i.e. targets that are always out of date
## This is done to ensure that the commands are always executed, even if a file with the same name exists
## See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
## Remove this if you want to use this Makefile for real targets
.PHONY: requirements data dev_requirements clean cover show_annotation clear_logs\
        clear_hyd tests main sglang_local sglang_cloud launch_llm hpc_job_test hpc_job\
				hpc_permissions zip_it build_documentation serve_documentation help main_zip_clear\
				clear_wandb

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = src
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
# create_environment:
#	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
# Needed for demjson. TODO: consider alternative to demjson
	- $(PYTHON_INTERPRETER) -m pip install "setuptools<58.0.0" 
	- $(PYTHON_INTERPRETER) -m pip install -r requirements.txt

dev_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install -e .["dev"]

# Ther is an issue with vllm==0.4.0.post1, force 0.3.3
# It installs triton 2.1.0, therefore vllm, then triton.
	- $(PYTHON_INTERPRETER) -m pip install -U vllm==0.3.3

# To launch a LLM server, triton 2.2.0 is required 
# It will trigger a dependency error with torch 2.1.2 that requires triton 2.1.0
# But so far it seems to work.
	- $(PYTHON_INTERPRETER) -m pip install -U triton==2.2.0

# To fine-tune model (NVIDIA A100)
	- $(PYTHON_INTERPRETER) -m pip install "unsloth[cu121_ampere_torch211] @ git+https://github.com/unslothai/unsloth.git"

# Used in SGLang runtime to accelerate attention computation
# Add --enable-flashinfer when launching the LLM server
# python -c "import torch; print(torch.version.cuda)" --> 12.1 	--> cu121
# pip list | grep torch --> torch 2.1.1 												--> torch2.1
	- $(PYTHON_INTERPRETER) -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.1/

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Get coverage report
cover:
	coverage run -m pytest tests/
	coverage report

## Clear logs
clear_logs:
	@rm -rf logs/*
	@rm -rf logs_dev/*

# Clear hyd
clear_hyd:
	@rm -rf hyd/*

# Clear wandb
clear_wandb:
	@rm -rf wandb/*

## Show annotations (using Streamlit)
show_annotation: clear_logs
	ENVIRONMENT="DEVELOPMENT" python -m streamlit run src/data/annotation_app.py	

# To be used in the future.
#run_dev:
#    ENVIRONMENT="DEVELOPMENT" python script.py
#
run:
	python src/train_model.py

## Run tests ("pytest -s tests/" to show the print statements)
tests:
	pytest tests/

sglang_local:
	python src/spe/sglang_m1.py -m llm_backend=local

sglang_cloud:
	python src/spe/sglang_m1.py -m llm_backend=cloud

launch_llm:
	$(eval LLM_PARAMETERS=$(shell python scripts/fetch_and_set_env_to_launch_llm.py))
	$(eval LLM_DIR=$(word 1, $(LLM_PARAMETERS)))
	$(eval LLM_MFS=$(word 2, $(LLM_PARAMETERS)))
	$(eval LLM_GPU_COUNT=$(word 3, $(LLM_PARAMETERS)))
	$(eval COMMAND=$(PYTHON_INTERPRETER) -m sglang.launch_server --model-path $(LLM_DIR) --tokenizer-path $(LLM_DIR) --port 30000 --enable-flashinfer --mem-fraction-static $(LLM_MFS) --tp $(LLM_GPU_COUNT))
	@echo "$(COMMAND)"
	@$(COMMAND)

main:
	$(MAKE) clear_logs
	python src/main.py

# Zip log files and latest hydra directory
zip_it:
	@echo "Deleting existing .zip files in experiments/ folder..."
	@LATEST_DIR=$$(find /home/evs/MT/hyd/hyd_run/ -type d -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2); \
	echo "LATEST_DIR: $$LATEST_DIR"; \
	TIMESTAMP=$$(date +"%b%d_%Hh%Mm%Ss"); \
	echo "TIMESTAMP: $$TIMESTAMP"; \
	echo "Zipping latest directory into hyd_$$TIMESTAMP.zip..."; \
	zip -r /home/evs/experiments/hyd_$$TIMESTAMP.zip $$LATEST_DIR; \
	echo "Zipping logs_dev into log_$$TIMESTAMP.zip..."; \
	zip -r /home/evs/experiments/log_$$TIMESTAMP.zip /home/evs/MT/logs_dev/*;\
	echo "Zipping wandb into wandb_$$TIMESTAMP.zip..."; \
	if [ -d /home/evs/MT/wandb ] && [ "$$(ls -A /home/evs/MT/wandb)" ]; then \
		zip -r /home/evs/experiments/wandb_$$TIMESTAMP.zip /home/evs/MT/wandb/*; \
	else \
		echo "/home/evs/MT/wandb does not exist, skipping..."; \
	fi


main_zip_clear:
	@$(MAKE) --no-print-directory main
	@$(MAKE) --no-print-directory zip_it
	@$(MAKE) --no-print-directory clear_hyd

# Set HPC job permissions
hpc_permissions:
	chmod +x hpc/jobs/dtu_job_emulate_llm.sh
	chmod +x hpc/jobs/dtu_job_llm.sh

# HPC command (test job)
hpc_job_test: hpc_permissions
	bsub < hpc/jobs/dtu_job_emulate_llm.sh

# HPC command (real job)
hpc_job: hpc_permissions
	bsub < hpc/jobs/dtu_job_llm.sh


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process raw data into processed data
data:
	python $(PROJECT_NAME)/data/make_dataset.py

#################################################################################
# Documentation RULES                                                           #
#################################################################################

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml	

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
