import pandas as pd
import sys
import string
import re

def replace(string, substitutions):
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)

######################Geneder################################

'''
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
'''

#####################Race#####################################

df_race = pd.read_csv('races.csv', header=None)
print(df_race.head(n=11))
df_list = df_race.T.values.tolist()
print(df_list)

black_bias_text = None

with open("race_black_bias_manual_swapped_attr_test.txt", 'r') as file:
    black_bias_text = file.readlines()
    file.close()

# white
'''substitutions = {str(b_word).lower(): str(df_list[1][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[1][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

white_bias_text = []
for b_sen in black_bias_text:
    w_sen = b_sen
    w_sen = replace(w_sen, merge_subs)
    white_bias_text.append(w_sen)

stdout = sys.stdout
with open("race_white_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in white_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout
'''
# Hispanic
'''substitutions = {str(b_word).lower(): str(df_list[2][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[2][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

hispanic_bias_text = []
for b_sen in black_bias_text:
    h_sen = b_sen
    h_sen = replace(h_sen, merge_subs)
    hispanic_bias_text.append(h_sen)

stdout = sys.stdout
with open("race_hispanic_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in hispanic_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout
'''
# Asian
'''substitutions = {str(b_word).lower(): str(df_list[3][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[3][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

asian_bias_text = []
for b_sen in black_bias_text:
    a_sen = b_sen
    a_sen = replace(a_sen, merge_subs)
    asian_bias_text.append(a_sen)

stdout = sys.stdout
with open("race_asian_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in asian_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout
'''

#Native american
substitutions = {str(b_word).lower(): str(df_list[4][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[4][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

native_bias_text = []
for b_sen in black_bias_text:
    n_sen = b_sen
    n_sen = replace(n_sen, merge_subs)
    native_bias_text.append(n_sen)

stdout = sys.stdout
with open("race_native_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in native_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout
