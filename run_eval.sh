#!/bin/bash
#SBATCH --job-name=eval-job-roberta8
#SBATCH --account=def-nidhih
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G        # memory per node
#SBATCH --time=00-01:00      # time (DD-HH:MM)
#SBATCH --output=/scratch/deep1401/debiasing_language_models/logs_eval/%N-%j.out  # %N for node name, %j for jobID

module load python/3.10.2 StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/8.0.0
virtualenv --no-download ENV
source ENV/bin/activate
pip install torch torchvision --no-index
pip install datasets transformers rich colorama joblib scikit-learn
echo ${BIAS}
echo ${DEBIAS_SIZE}
echo ${FRACTION_SIZE}
echo ${SAVEDMODELPATH}
echo ${SAVEDOUTPUTPATH}
# cd src && python main.py ${BIAS} ${DEBIAS_SIZE} ${FRACTION_SIZE}

cd evaluation && CUDA_LAUNCH_BLOCKING=1 python eval_discriminative_models.py --pretrained-class ${SAVEDMODELPATH} --input-file data/stereoset/dev.json --output-dir ${SAVEDOUTPUTPATH} --output-file output.json --intrasentence-model RoBERTaLM --intersentence-model ModelNSP --tokenizer RobertaTokenizer && 
python evaluation.py --gold-file data/stereoset/dev.json --predictions-file ${SAVEDOUTPUTPATH}/output.json --output-file ${SAVEDOUTPUTPATH}/output.txt
