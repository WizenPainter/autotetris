# !/bin/sh
### General options
### –- specify queue --
BSUB -q gpua100 #SELECT YOUR GPU QUEUE
### -- set the job Name --
BSUB -J SGDTest
### -- ask for number of cores (default: 1) --
# BSUB -n 8 #Number of cores (4/gpu)
# BSUB -R "span[hosts=1]" #Always set this to 1
## -- Select the resources: 1 gpu in exclusive process mode --
# BSUB -gpu "num=1:mode=exclusive_process" #Number of GPUs requested
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
# BSUB -W 23:59 #Walltime needed for computation (Be sure to set a lower time, if you want to have your job run faster)
# request 5GB of system-memory
# BSUB -R "rusage[mem=8GB]" #RAM Required for computation (per CPU, careful about modifying this)
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
BSUB -u <s22025@dtu.dk>
### -- send notification at start --
BSUB -B
### -- send notification at completion--
BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
# BSUB -o gpu-%J.out
# BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi

# Load the relevant CUDA module (Only necessary on a100s)
module load cuda/11.3


# Make sure to change directory to the directory of your project
#cd /work3/sXXXXXX/auto-tetris 

# Make sure to load your environment
source .venv/bin/activate

python3 model_training_hpc.py