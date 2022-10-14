#!/bin/bash
declare -a BiasSet=("none")
declare -a DebiasSet=("0")
declare -a FractionSet=("0.01" "0.02" "0.04" "0.08" "0.16" "0.32")
for bias in ${BiasSet[@]};
	do
	for debias_size in ${DebiasSet[@]};
		do
		for fraction_size in ${FractionSet[@]};
			do
			echo ${bias}
			echo ${debias_size}
			echo ${fraction_size}
			sbatch --export=BIAS=${bias},DEBIAS_SIZE=${debias_size},FRACTION_SIZE=${fraction_size} run.sh
			done
		done
	done
