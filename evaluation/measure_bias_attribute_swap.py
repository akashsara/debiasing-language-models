"""
This script performs Student t-test on the perplexity distribution of two sentences groups with contrasting attributes
"""
import pandas as pd
import numpy as np
from scipy import stats
import helper_functions as helpers
import time
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from transformers import AutoModelWithLMHead, AutoTokenizer, T5Model, T5ForConditionalGeneration
import torch
from transformers import T5Tokenizer

model_params = {
    "OUTPUT_PATH": "../models",  # output path
    "MODEL": "../models/religion_model/model_files/",  # model_type: t5-base/t5-large
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 32,  # max length of target text
    "EARLY_STOPPING_PATIENCE": 1,  # number of epochs before stopping training.
    "SENTINEL_MASK_FRACTION": 0.15,  # Fraction of a sequence to sentinel mask
    "BATCH_SIZE": 32,  # Batch size to use
    "WORD_LIST": "../data/religion.csv",
    "REGULARISATION_LAMBDA": 0.1
}

torch.manual_seed(0)


def get_perplexity_list(df, m, t):
    """
        Gets perplexities of all sentences in a DataFrame based on given model
        Parameters
        ----------
        df : pd.DataFrame
        DataFrame with Reddit comments
        m : model
        Pre-trained language model
        t : tokenizer
        Pre-trained tokenizer for the given model

        Returns
        -------
        List of sentence perplexities
    """
    perplexity_list = []
    for idx, row in enumerate(df[0].values.tolist()):
        try:
            perplexity = helpers.perplexity_score(row, m, t)
        except Exception as ex:
            print(ex.__repr__())
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_perplexity_list_test(df, m, t, dem):
    """
    Gets perplexities of all sentences in a DataFrame(contains 2 columns of contrasting sentences) based on given model
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments in 2 columns
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    List of sentence perplexities
    """
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            if dem == 'black':
                perplexity = helpers.perplexity_score(row, m, t)
            else:
                perplexity = helpers.perplexity_score(row, m, t)
        except Exception as ex:
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_model_perplexity(df, m, t):
    """
    Finds model perplexity based on average model loss over all sentences
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    Model perplexity
    """
    model_perplexity = helpers.model_perplexity(df, m, t)
    return model_perplexity


def find_anomalies(data):
    """
    Find outliers in a given data distribution
    Parameters
    ----------
    data : list
    List of sentence perplexities

    Returns
    -------
    List of outliers
    """
    anomalies = []

    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies


# --------------------------- Religion ---------------------------------------

# calculation of perplexity of religion
'''
tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])


islam_df = pd.read_csv('../data/religion_islam_bias_manual_swapped_attr_test.txt', header=None)
taoism_df = pd.read_csv('../data/religion_taoism_bias_manual_swapped_attr_test.txt', header=None)

#print(islam_df.head(n=10))
#print(taoism_df.head(n=10))

islam_perplexity = get_perplexity_list(islam_df, model, tokenizer)
taoism_perplexity = get_perplexity_list(taoism_df, model, tokenizer)

islam_df['perplexity'] = islam_perplexity
taoism_df['perplexity'] = taoism_perplexity

print('Instances in demo 1 and 2: {}, {}'.format(len(islam_perplexity), len(taoism_perplexity)))

print('Mean and Std of unfiltered perplexities Islam - Mean {}, Variance {}'.format(np.mean(islam_perplexity), np.std(islam_perplexity)))
print('Mean and Std of unfiltered perplexities taoism - Mean {}, Variance {}'.format(np.mean(taoism_perplexity), np.std(taoism_perplexity)))

assert len(islam_perplexity) == len(taoism_perplexity)

demo1_out = find_anomalies(np.array(islam_perplexity))
demo2_out = find_anomalies(np.array(taoism_perplexity))

print(demo1_out, demo2_out)
demo1_in = [d1 for d1 in islam_perplexity if d1 not in demo1_out]
demo2_in = [d2 for d2 in taoism_perplexity if d2 not in demo2_out]

for i, (p1, p2) in enumerate(zip(islam_perplexity, taoism_perplexity)):
    if p1 in demo1_out or p2 in demo2_out:
        islam_df.drop(islam_df.loc[islam_df['perplexity'] == p1].index, inplace=True)
        taoism_df.drop(taoism_df.loc[taoism_df['perplexity'] == p2].index, inplace=True)

print('Mean and Std of filtered perplexities islam - Mean {}, Variance {}'.format(np.mean(islam_df['perplexity']), np.std(islam_df['perplexity'])))
print('Mean and Std of filtered perplexities taoism - Mean {}, Variance {}'.format(np.mean(taoism_df['perplexity']), np.std(taoism_df['perplexity'])))

t_value, p_value = stats.ttest_ind(islam_perplexity, taoism_perplexity, equal_var=False)

print('Unfiltered perplexities - T value {} and P value {}'.format(t_value, p_value))
#print(t_value, p_value)

print("Length after outlier removal: ", len(islam_df['perplexity']), len(taoism_df['perplexity']))
t_unpaired, p_unpaired = stats.ttest_ind(islam_df['perplexity'].to_list(), taoism_df['perplexity'].to_list(), equal_var=False)
print('Student(unpaired) t-test, after outlier removal: t-value {}, p-value {}'.format(t_unpaired, p_unpaired))

t_paired, p_paired = stats.ttest_rel(islam_df['perplexity'].to_list(), taoism_df['perplexity'].to_list())
print('Paired t-test, after outlier removal: t-value {}, p-value {}'.format(t_paired, p_paired))
'''


# ----------------------- Gender ------------------------------------------------

pretrained_gender_model = '../models/gender_model/'

tokenizer = T5Tokenizer.from_pretrained(pretrained_gender_model)
model = T5ForConditionalGeneration.from_pretrained(pretrained_gender_model)


female_df = pd.read_csv('../data/gender_female_bias_manual_swapped_attr_test.txt', header=None)
male_df = pd.read_csv('../data/gender_male_bias_manual_swapped_attr_test.txt', header=None)

#print(female_df.head(n=10))
#print(male_df.head(n=10))

female_perplexity = get_perplexity_list(female_df, model, tokenizer)
male_perplexity = get_perplexity_list(male_df, model, tokenizer)

female_df['perplexity'] = female_perplexity
male_df['perplexity'] = male_perplexity

print('Instances in demo 1 and 2: {}, {}'.format(len(female_perplexity), len(male_perplexity)))

print('Mean and Std of unfiltered perplexities female - Mean {}, Variance {}'.format(np.mean(female_perplexity), np.std(female_perplexity)))
print('Mean and Std of unfiltered perplexities male - Mean {}, Variance {}'.format(np.mean(male_perplexity), np.std(male_perplexity)))

assert len(female_perplexity) == len(male_perplexity)

demo1_out = find_anomalies(np.array(female_perplexity))
demo2_out = find_anomalies(np.array(male_perplexity))

print(demo1_out, demo2_out)
demo1_in = [d1 for d1 in female_perplexity if d1 not in demo1_out]
demo2_in = [d2 for d2 in male_perplexity if d2 not in demo2_out]

for i, (p1, p2) in enumerate(zip(female_perplexity, male_perplexity)):
    if p1 in demo1_out or p2 in demo2_out:
        female_df.drop(female_df.loc[female_df['perplexity'] == p1].index, inplace=True)
        male_df.drop(male_df.loc[male_df['perplexity'] == p2].index, inplace=True)

print('Mean and Std of filtered perplexities female - Mean {}, Variance {}'.format(np.mean(female_df['perplexity']), np.std(female_df['perplexity'])))
print('Mean and Std of filtered perplexities male - Mean {}, Variance {}'.format(np.mean(male_df['perplexity']), np.std(male_df['perplexity'])))

t_value, p_value = stats.ttest_ind(female_perplexity, male_perplexity, equal_var=False)

print('Unfiltered perplexities - T value {} and P value {}'.format(t_value, p_value))
#print(t_value, p_value)

print("Length after outlier removal: ", len(female_df['perplexity']), len(male_df['perplexity']))
t_unpaired, p_unpaired = stats.ttest_ind(female_df['perplexity'].to_list(), male_df['perplexity'].to_list(), equal_var=False)
print('Student(unpaired) t-test, after outlier removal: t-value {}, p-value {}'.format(t_unpaired, p_unpaired))

t_paired, p_paired = stats.ttest_rel(female_df['perplexity'].to_list(), male_df['perplexity'].to_list())
print('Paired t-test, after outlier removal: t-value {}, p-value {}'.format(t_paired, p_paired))
