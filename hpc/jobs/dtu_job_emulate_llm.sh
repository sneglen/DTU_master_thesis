#!/bin/sh
### General options
### -- specify queue -- (hpc, gpuv100:https://www.hpc.dtu.dk/?page_id=2759, gpuv100i:used in voltash)
#BSUB -q gpuv100
### -- set job Name --
#BSUB -J LLM_emulation
### -- request quantity of cores (default: 1) --
#BSUB -n 4
### -- request that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- request 4GB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
### -- request to kill job if it exceeds 5 GB per core/slot --
#BSUB -M 5GB
### -- Select GPU resources --- (e.g: 2xGPU shared: "num=2:mode=shared", 1xGPU exclusive: "num=1:mode=exclusive_process")
#BSUB -gpu "num=1:mode=shared"  
### -- walltime limit: hh:mm --
#BSUB -W 00:05
### -- email address --
#BSUB -u my@email.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- specify the output and error file. %J is the job-id --
#BSUB -o hpc/out/%J_job.out
#BSUB -e hpc/out/%J_job.err

### python output --> stdout = 1
### echo          --> stderr = 2

echo "JOB: Loading modules..." >&2
nvidia-smi
module load python3/3.11.7
module load cuda/12.1

# Debugging CUDA "not found" error
echo "DEBUG: CUDA_HOME: $CUDA_HOME" >&2
echo "DEBUG: PATH: $PATH" >&2
echo "DEBUG: LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >&2

# Activate env_MT virtual environment (just in case):
echo "JOB: Activating environment..." >&2
source /zhome/ac/8/105765/venv/env_MT/bin/activate

# Start LLM server in the background
echo "JOB: Starting LLM server..." >&2
python hpc/test_code/dtu_emulate_llm_server.py > hpc/out/${LSB_JOBID}_llm_server.out 2>&1 &
SERVER_PID=$!

# Wait for the server to initialize (time in seconds)
sleep 10

echo "JOB: Starting LLM client..." >&2
python hpc/test_code/dtu_emulate_llm_client.py > hpc/out/${LSB_JOBID}_llm_client.out 2>&1

echo "JOB: Shutting down (kill if possible)..." >&2
if kill -0 $SERVER_PID 2>/dev/null; then
    kill $SERVER_PID
else
    echo "JOB: LLM server process ($SERVER_PID) not found. It may have already terminated." >&2
fi





















