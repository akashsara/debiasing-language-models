#!/bin/bash
declare -a BiasSet=("gender" "race" "religion")
declare -a DebiasSetGender=("500" "1000")
declare -a DebiasSetRace=("500" "1000" "2000" "4000" "8000")
declare -a DebiasSetReligion=("500" "1000" "2000" "4000" "8000" "16000")
declare -a FractionSet=("0.01" "0.02" "0.04" "0.08" "0.16" "0.32")
modelclass="models_roberta_int_8"

for debias_size in ${DebiasSetGender[@]};
	do
	for fraction_size in ${FractionSet[@]};
		do
		bias="gender"
		modelpath="/home/deep1401/scratch/debiasing_language_models/$modelclass/downstream/$bias/new_debsize_$debias_size/lm_$fraction_size/model_files"
		outputpath="output_files/$modelclass/$bias/new_debsize_$debias_size/lm_$fraction_size"
		echo ${bias}
		echo ${debias_size}
		echo ${fraction_size}
		sbatch --export=BIAS=${bias},DEBIAS_SIZE=${debias_size},FRACTION_SIZE=${fraction_size},SAVEDMODELPATH=${modelpath},SAVEDOUTPUTPATH=${outputpath} run_eval.sh
		done
	done


for debias_size in ${DebiasSetRace[@]};
	do
	for fraction_size in ${FractionSet[@]};
		do
		bias="race"
		modelpath="/home/deep1401/scratch/debiasing_language_models/$modelclass/downstream/${bias}/new_debsize_$debias_size/lm_$fraction_size/model_files"
		outputpath="output_files/$modelclass/$bias/new_debsize_$debias_size/lm_$fraction_size"
		echo ${bias}
		echo ${debias_size}
		echo ${fraction_size}
		sbatch --export=BIAS=${bias},DEBIAS_SIZE=${debias_size},FRACTION_SIZE=${fraction_size},SAVEDMODELPATH=${modelpath},SAVEDOUTPUTPATH=${outputpath} run_eval.sh
		done
	done


for debias_size in ${DebiasSetReligion[@]};
	do
	for fraction_size in ${FractionSet[@]};
		do
		bias="religion"
		modelpath="/home/deep1401/scratch/debiasing_language_models/$modelclass/downstream/$bias/new_debsize_$debias_size/lm_$fraction_size/model_files"
		outputpath="output_files/$modelclass/$bias/new_debsize_$debias_size/lm_$fraction_size"
		echo ${bias}
		echo ${debias_size}
		echo ${fraction_size}
		sbatch --export=BIAS=${bias},DEBIAS_SIZE=${debias_size},FRACTION_SIZE=${fraction_size},SAVEDMODELPATH=${modelpath},SAVEDOUTPUTPATH=${outputpath} run_eval.sh
		done
	done

