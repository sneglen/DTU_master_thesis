FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Working directory in container
WORKDIR /app

# Install Python 3.10 (and 'make')
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    make \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3.10 -m pip install --upgrade pip


# Install production and development dependencies
RUN pip install -r requirements.txt
RUN pip install -r requirements_dev.txt

# Copy application's code into container BEFORE "pip install -e .["dev"]"
COPY . /app

RUN pip install -e .["dev"]
RUN pip install -U vllm==0.3.3
RUN pip install -U triton==2.2.0


# CMD ["python3.10", "whateverfile.py"]
CMD ["make", "tests"]