import pandas as pd
import sys
import string
import re
import inflect

infl = inflect.engine()

# remove punctuations from a sentence
def remove_punctuations(sen):
    return sen.translate(str.maketrans('', '', string.punctuation))


# Replace words of a sentence with new words

def replace(sentence, substitions):
    sentence = sentence.lower()
    words = remove_punctuations(sentence).split(' ')
    try:
        words = compressx(words.index("saudi"), words.index("saudi")+2, words)
#        print(sen_a)
    except:
        print()
    flag = True
    for i in range(len(words)):
        if words[i] in substitions and flag:
            words[i] = substitions[words[i]]
            flag = False
    return ' '.join(words)

#replace_sentence("He is a very good boy. His mother is a doctor.", {'he':'she', 'his':'hers', 'boy':'girl'})


# def replace(string, substitutions):
#     substrings = sorted(substitutions, key=len, reverse=True)
#     regex = re.compile('|'.join(map(re.escape, substrings)))
#     return regex.sub(lambda match: substitutions[match.group(0)], string)

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

'''df_race = pd.read_csv('races.csv', header=None)
print(df_race.head(n=11))
df_list = df_race.T.values.tolist()
print(df_list)

black_bias_text = None

with open("race_black_bias_manual_swapped_attr_test.txt", 'r') as file:
    black_bias_text = file.readlines()
    file.close()
'''
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
'''substitutions = {str(b_word).lower(): str(df_list[4][i]).lower() for i, b_word in enumerate(df_list[0])}
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
'''
############################ Religion ################################

'''df_religion = pd.read_csv('religion.csv')
print(df_religion.head(n=11))
df_list = df_religion.T.values.tolist()
print(df_list)

islam_bias_text = None

with open("religion_islam_bias_manual_swapped_attr_test.txt", 'r') as file:
    islam_bias_text = file.readlines()
    file.close()'''

# Christianity
'''substitutions = {str(b_word).lower(): str(df_list[1][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[1][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

christianity_bias_text = []
for i_sen in islam_bias_text:
    c_sen = i_sen
    c_sen = replace(c_sen, merge_subs)
    christianity_bias_text.append(c_sen)

stdout = sys.stdout
with open("religion_christianity_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in christianity_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout'''

# Judaism
'''substitutions = {str(b_word).lower(): str(df_list[2][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[2][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

judaism_bias_text = []
for i_sen in islam_bias_text:
    c_sen = i_sen
    c_sen = replace(c_sen, merge_subs)
    judaism_bias_text.append(c_sen)

stdout = sys.stdout
with open("religion_judaism_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in judaism_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout'''

# Hinduism

'''substitutions = {str(b_word).lower(): str(df_list[3][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[3][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

hinduism_bias_text = []
for i_sen in islam_bias_text:
    c_sen = i_sen
    c_sen = replace(c_sen, merge_subs)
    hinduism_bias_text.append(c_sen)

stdout = sys.stdout
with open("religion_hinduism_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in hinduism_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout'''

# Buddhism

'''substitutions = {str(b_word).lower(): str(df_list[4][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[4][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

buddhism_bias_text = []
for i_sen in islam_bias_text:
    c_sen = i_sen
    c_sen = replace(c_sen, merge_subs)
    buddhism_bias_text.append(c_sen)

stdout = sys.stdout
with open("religion_buddhism_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in buddhism_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout'''


# Confucianism

'''substitutions = {str(b_word).lower(): str(df_list[5][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[5][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

confucianism_bias_text = []
for i_sen in islam_bias_text:
    c_sen = i_sen
    c_sen = replace(c_sen, merge_subs)
    confucianism_bias_text.append(c_sen)

stdout = sys.stdout
with open("religion_confucianism_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in confucianism_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout'''


# Taoism

'''substitutions = {str(b_word).lower(): str(df_list[6][i]).lower() for i, b_word in enumerate(df_list[0])}
op_substitutions = {str(df_list[6][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
merge_subs = {**substitutions, **op_substitutions}

print(merge_subs)

taoism_bias_text = []
for i_sen in islam_bias_text:
    c_sen = i_sen
    c_sen = replace(c_sen, merge_subs)
    taoism_bias_text.append(c_sen)

stdout = sys.stdout
with open("religion_taoism_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in taoism_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout'''

def compressx(min_index = -1, max_index = -1, x = None):
    if max_index < len(x):
        x = x[:min_index] + ['-'.join(x[min_index:max_index])] + x[max_index:]
    else:
        x = x[:min_index] + ['-'.join(x[min_index:max_index])]
    return x

def diff_words(sen_a, sen_b):
    sen_a = sen_a.split(' ')
    sen_b = sen_b.split(' ')
    try:
        sen_a = compressx(sen_a.index("saudi"), sen_a.index("saudi")+2, sen_a)
#        print(sen_a)
    except:
        print()
    
    try:
        sen_b = compressx(sen_b.index("saudi"), sen_b.index("saudi")+2, sen_b)
 #       print(sen_b)
    except:
        print()

    words = []
    for i in range(len(sen_a)):
        try:
            if sen_a[i] != sen_b[i]:
                words.append(sen_b[i].translate(str.maketrans('', '', string.punctuation)))
        except:
            print("Exception:")
            #print(sen_a)
            #print(sen_b)
    return words

#####################################Columned Based Data Preparation#############################################

df_religion = pd.read_csv('word_lists/religion.csv')
print(df_religion.head(n=11))
df_list = df_religion.T.values.tolist()
print(df_list)

islam_bias_text = None

with open("data/religion_islam_bias_manual_swapped_attr_test.txt", 'r') as file:
    islam_bias_text = file.readlines()
    file.close()

column_combined_biased_text = []

base_text = []
first_item_flag = False

for rel in range(1, 7):
    substitutions = {str(b_word).lower(): str(df_list[rel][i]).lower() for i, b_word in enumerate(df_list[0])}
    op_substitutions = {str(df_list[rel][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
    merge_subs = {**substitutions, **op_substitutions}
    merge_subs.update({**op_substitutions, **substitutions})
    
    substitutions = {infl.plural(str(b_word)).lower(): infl.plural(str(df_list[rel][i])).lower() for i, b_word in enumerate(df_list[0])}
    op_substitutions = {infl.plural(str(df_list[rel][i])).lower(): infl.plural(str(b_word)).lower() for i, b_word in enumerate(df_list[0])}
    merge_subs.update({**substitutions, **op_substitutions})
    merge_subs.update({**op_substitutions, **substitutions})    

    print(merge_subs)

    biased_text = []

    for i_sen in islam_bias_text:
        i_sen = i_sen.strip()
        c_sen = i_sen
        c_sen = replace(c_sen, merge_subs)
        biased_text.append((c_sen, diff_words(remove_punctuations(i_sen), c_sen)))
        if first_item_flag == False:
            base_text.append((remove_punctuations(i_sen), diff_words(c_sen, remove_punctuations(i_sen))))
    if first_item_flag == False:
        column_combined_biased_text.append(base_text)
    first_item_flag = True
    column_combined_biased_text.append(biased_text)
#print(column_combined_biased_text)
df = pd.DataFrame(column_combined_biased_text)
df = df.T
print(df)
df.to_csv('data/column_based_religion_data.csv', header = list(df_religion.columns.values), index=False)

'''

stdout = sys.stdout
with open("religion_christianity_bias_manual_swapped_attr_test.txt", 'w') as file:
    sys.stdout = file
    for sen in christianity_bias_text:
        print(sen.replace('\n', ''))
    file.close()
sys.stdout = stdout
'''


############################### Gender ###################################
df_religion = pd.read_csv('word_lists/gender.csv')
print(df_religion.head(n=11))
df_list = df_religion.T.values.tolist()
print(df_list)

islam_bias_text = None

with open("data/gender_male_bias_manual_swapped_attr_test.txt", 'r') as file:
    islam_bias_text = file.readlines()
    file.close()

column_combined_biased_text = []

base_text = []
first_item_flag = False

for rel in range(1, 2):
    substitutions = {str(b_word).lower(): str(df_list[rel][i]).lower() for i, b_word in enumerate(df_list[0])}
    op_substitutions = {str(df_list[rel][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
    merge_subs = {**substitutions, **op_substitutions}
    merge_subs.update({**op_substitutions, **substitutions})

    substitutions = {infl.plural(str(b_word)).lower(): infl.plural(str(df_list[rel][i])).lower() for i, b_word in enumerate(df_list[0])}
    op_substitutions = {infl.plural(str(df_list[rel][i])).lower(): infl.plural(str(b_word)).lower() for i, b_word in enumerate(df_list[0])}
    merge_subs.update({**substitutions, **op_substitutions})
    merge_subs.update({**op_substitutions, **substitutions})
     
    print(merge_subs)

    biased_text = []

    for i_sen in islam_bias_text:
        i_sen = i_sen.strip()
        c_sen = i_sen
        c_sen = replace(c_sen, merge_subs)
        biased_text.append((c_sen, diff_words(remove_punctuations(i_sen), c_sen)))
        if first_item_flag == False:
            base_text.append((remove_punctuations(i_sen), diff_words(c_sen, remove_punctuations(i_sen))))
    if first_item_flag == False:
        column_combined_biased_text.append(base_text)
    first_item_flag = True
    column_combined_biased_text.append(biased_text)
#print(column_combined_biased_text)
df = pd.DataFrame(column_combined_biased_text)
df = df.T
print(df)
df.to_csv('data/column_based_gender_data.csv', header = list(df_religion.columns.values), index=False)


#####################################Columned Based Data Preparation#############################################

df_religion = pd.read_csv('word_lists/races.csv')
print(df_religion.head(n=11))
df_list = df_religion.T.values.tolist()
print(df_list)

islam_bias_text = None

with open("data/race_black_bias_manual_swapped_attr_test.txt", 'r') as file:
    islam_bias_text = file.readlines()
    file.close()

column_combined_biased_text = []

base_text = []
first_item_flag = False

for rel in range(1, 5):
    substitutions = {str(b_word).lower(): str(df_list[rel][i]).lower() for i, b_word in enumerate(df_list[0])}
    op_substitutions = {str(df_list[rel][i]).lower(): str(b_word).lower() for i, b_word in enumerate(df_list[0])}
    merge_subs = {**substitutions, **op_substitutions}
    merge_subs.update({**op_substitutions, **substitutions})
    
    substitutions = {infl.plural(str(b_word)).lower(): infl.plural(str(df_list[rel][i])).lower() for i, b_word in enumerate(df_list[0])}
    op_substitutions = {infl.plural(str(df_list[rel][i])).lower(): infl.plural(str(b_word)).lower() for i, b_word in enumerate(df_list[0])}
    merge_subs.update({**substitutions, **op_substitutions})
    merge_subs.update({**op_substitutions, **substitutions})
    

    print(merge_subs)

    biased_text = []

    for i_sen in islam_bias_text:
        i_sen = i_sen.strip()
        c_sen = i_sen
        c_sen = replace(c_sen, merge_subs)
        biased_text.append((c_sen, diff_words(remove_punctuations(i_sen), c_sen)))
        if first_item_flag == False:
            base_text.append((remove_punctuations(i_sen), diff_words(c_sen, remove_punctuations(i_sen))))
    if first_item_flag == False:
        column_combined_biased_text.append(base_text)
    first_item_flag = True
    column_combined_biased_text.append(biased_text)
#print(column_combined_biased_text)
df = pd.DataFrame(column_combined_biased_text)
df = df.T
print(df)
df.to_csv('data/column_based_race_data.csv', header = list(df_religion.columns.values), index=False)

