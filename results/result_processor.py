import sys
import pandas as pd

religion_list = ["christianity", "judaism", "hinduism", "buddhism", "confucianism", "taoism"]
gender_list = ["male"]
race_list = ["white", "native", "asian", "hispanic"]

# -------------------------------- Result Processing for t_score

'''religion_regu_t_values = []
religion_base_t_values = []

for r in religion_list:
    with open("../results/IslamVs{}Results.txt".format(r), 'r') as file:
        t_val = float(((str((file.readlines())[-1]).split(' ')[6]).replace(',','')))
        religion_regu_t_values.append(t_val)
        file.close()
    
    with open("../results/Base_IslamVs{}Results.txt".format(r), 'r') as file:
        t_val = float(((str((file.readlines())[-1]).split(' ')[6]).replace(',','')))
        religion_base_t_values.append(t_val)
        file.close()

gender_regu_t_values = []
gender_base_t_values = []

for r in gender_list:
    with open("../results/FemaleVs{}Results.txt".format(r), 'r') as file:
        t_val = float(((str((file.readlines())[-1]).split(' ')[6]).replace(',', '')))
        gender_regu_t_values.append(t_val)
        file.close()

    with open("../results/Base_FemaleVs{}Results.txt".format(r), 'r') as file:
        t_val = float(((str((file.readlines())[-1]).split(' ')[6]).replace(',', '')))
        gender_base_t_values.append(t_val)
        file.close()

race_regu_t_values = []
race_base_t_values = []

for r in race_list:
    with open("../results/BlackVs{}Results.txt".format(r), 'r') as file:
        t_val = float(((str((file.readlines())[-1]).split(' ')[6]).replace(',', '')))
        race_regu_t_values.append(t_val)
        file.close()

    with open("../results/Base_BlackVs{}Results.txt".format(r), 'r') as file:
        t_val = float(((str((file.readlines())[-1]).split(' ')[6]).replace(',', '')))
        race_base_t_values.append(t_val)
        file.close()

with open("../results/t_score_table.txt", 'w') as file:
    sys.stdout = file
    print("Class\t&\t Regularized Model\t & \t Unregularized Model\\\\")

    for i in range(len(religion_list)):
        print("{}\t&\t {} \t & \t {}\\\\".format("Islam Vs "+religion_list[i].capitalize(), religion_regu_t_values[i], religion_base_t_values[i]))

    for i in range(len(gender_list)):
        print("{}\t&\t  {} \t & \t {}\\\\".format("Female Vs "+gender_list[i].capitalize(), religion_regu_t_values[i], gender_base_t_values[i]))

    for i in range(len(race_list)):
        print("{}\t&\t  {} \t & \t {}\\\\".format("Black Vs "+race_list[i].capitalize(), religion_regu_t_values[i], race_base_t_values[i]))


'''

# ---------------------- Result processing for different Lambda Values ------------------------------

religion_regu_t_values = []
gender_regu_t_values = []
race_regu_t_values = []

lambda_list = ['0.01','0.2','0.5','1']

for lm in lambda_list:
    for r in religion_list:
        with open("../results/IslamVs{}Results_{}.txt".format(r, lm), 'r') as file:
            t_val = float(((str((file.readlines())[-1]).split(' ')[6]).replace(',', '')))
            religion_regu_t_values.append(t_val)
            file.close()

with open("../results/t_score_table_lambda.txt", 'w') as file:
    sys.stdout = file
    print("Class\t& 0.01 \t& 0.2\t & 0.5 \t & 1.0\\\\")

    for i in range(len(religion_list)):
        print("{}".format(religion_list[i].capitalize()))
        for j in range(len(lambda_list)):
            print("\t& {}".format(religion_regu_t_values[j]))
        print("\\\\")
