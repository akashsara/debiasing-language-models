#!/bin/bash
#SBATCH --job-name=distilbert-int-4
#SBATCH --account=def-nidhih
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G        # memory per node
#SBATCH --time=00-20:00      # time (DD-HH:MM)
#SBATCH --output=/scratch/deep1401/debiasing_language_models/logs/%N-%j.out  # %N for node name, %j for jobID

module load python/3.10.2 StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/8.0.0
virtualenv --no-download ENV
source ENV/bin/activate
pip install torch torchvision --no-index
pip install datasets transformers rich colorama joblib scikit-learn
echo ${BIAS}
echo ${DEBIAS_SIZE}
echo ${FRACTION_SIZE}
cd src && python main.py ${BIAS} ${DEBIAS_SIZE} ${FRACTION_SIZE}

#cd evaluation && python evaluation.py --gold-file data/stereoset/dev.json --predictions-file output_files/output_1.json > output_final.txt
#cd evaluation && CUDA_LAUNCH_BLOCKING=1 python eval_discriminative_models.py --pretrained-class "/home/deep1401/scratch/debiasing_language_models/models_roberta_no_int/downstream/gender/new_debsize_1000/lm_0.32/model_files" --input-file data/stereoset/dev.json --output-dir output_files --output-file output_int1.json --intrasentence-model RoBERTaLM --intersentence-model ModelNSP --tokenizer RobertaTokenizer

#cd evaluation && python evaluation.py --gold-file data/stereoset/dev.json --predictions-file output_files/output_int1.json > output_files/output_int1.txt
