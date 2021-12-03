import datasets
import pandas as pd
import re

PREDICTIONS_FILE = '/Users/dhruvmullick/Projects/debiasing-language-models/models/predictions.csv'


def preprocess_each_line(line):
    line = line.replace('<pad>', '')
    line = line.replace(',', '')
    line = line.replace('.', '')
    line = line.replace('?', '')
    line = line.replace('!', '')
    line = line.replace('"', '')
    line = line.replace('</s>', '')
    line = line.strip()
    line = re.split('<extra_id_\d+>', line)
    line = [x for x in line if x]
    return line


def preprocess_dataframe(df):
    df['Generated Text'] = df['Generated Text'].apply(preprocess_each_line)
    df['Actual Text'] = df['Actual Text'].apply(preprocess_each_line)
    return df


with open(PREDICTIONS_FILE, 'r') as file_open:
    df = pd.read_csv(file_open)
    preprocess_dataframe(df)
    generated_terms = []
    actual_terms = []
    for idx, generated_list_one_case in enumerate(df['Generated Text']):
        actual_list_one_case = df['Actual Text'][idx]
        len_to_check = min(len(generated_list_one_case), len(actual_list_one_case))
        generated_list_one_case = generated_list_one_case[:len_to_check]
        actual_list_one_case = actual_list_one_case[:len_to_check]
        generated_terms.extend(generated_list_one_case)
        actual_terms.extend(actual_list_one_case)

    assert (len(generated_terms) == len(actual_terms))
    generated_terms = [x.strip() for x in generated_terms]
    actual_terms = [x.strip() for x in actual_terms]
    actual_terms_as_list = [[x] for x in actual_terms]

    #### Sacre BLEU
    metric = datasets.load_metric('sacrebleu')
    final_score = metric.compute(predictions=generated_terms, references=actual_terms_as_list)
    print(final_score)

    #### METEOR
    metric = datasets.load_metric('meteor')
    final_score = metric.compute(predictions=generated_terms, references=actual_terms)
    print(final_score)

    #### ROUGE
    metric = datasets.load_metric('rouge')
    final_score = metric.compute(predictions=generated_terms, references=actual_terms)
    print(final_score)


