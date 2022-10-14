#!/bin/bash
#SBATCH --account=def-nidhih
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G        # memory per node
#SBATCH --time=00-23:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load python/3.10.2 StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/8.0.0
virtualenv --no-download ENV
source ENV/bin/activate
pip install torch torchvision --no-index
pip install datasets transformers rich
echo ${BIAS}
echo ${DEBIAS_SIZE}
echo ${FRACTION_SIZE}
cd src && python main.py ${BIAS} ${DEBIAS_SIZE} ${FRACTION_SIZE}
