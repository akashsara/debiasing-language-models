import pandas as pd
import sys

######################Geneder################################

df_gender = pd.read_csv('gender.csv', encoding='utf-8')
print(df_gender.head(n=1))
df_list = df_gender.values.tolist()

female_bias_text = None

with open("gender_female_bias_manual_swapped_attr_test.txt", 'r') as file:
    female_bias_text = file.readlines()
    file.close()


male_bias_text = []

for f_sen in female_bias_text:
    m_sen = f_sen
    for pairs in df_list:
        m_sen = m_sen.replace(pairs[1], ' ' + pairs[0])
    male_bias_text.append(m_sen)

stdout = sys.stdout
with open("gender_male_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in male_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout


#####################
