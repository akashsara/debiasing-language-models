"""
Reference: https://github.com/SoumyaBarikeri/RedditBias/tree/master/Evaluation
This script performs Student t-test on the perplexity distribution of two sentences groups with contrasting attributes
"""
import pandas as pd
import numpy as np
from scipy import stats
import helper_functions as helpers
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
import logging
import sys
from transformers import AutoModelWithLMHead, AutoTokenizer, T5Model, T5ForConditionalGeneration
import torch
from transformers import T5Tokenizer

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
    model_perplexity = helpers.model_perplexity(df[0].values, m, t)
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

'''lambda_list = ["0.01", "0.2", "0.5", "1"]
for lm in lambda_list:
    pretrained_model = "../models/religion_with_reg/{}/model_files/".format(lm)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)

    religion_list = ["islam", "christianity", "judaism", "hinduism", "buddhism", "confucianism", "taoism"]

    for index in range(1, len(religion_list)):
        islam_df = pd.read_csv('../data/religion_islam_bias_manual_swapped_attr_test.txt', header=None)
        with open('../results/IslamVs{}Results_{}.txt'.format(religion_list[index], lm), 'w') as file:
            sys.stdout = file
            target_demo_df = pd.read_csv('../data/religion_{}_bias_manual_swapped_attr_test.txt'.format(religion_list[index]), header=None)

            #print(islam_df.head(n=10))
            #print(target_demo_df.head(n=10))

            islam_perplexity = get_perplexity_list(islam_df, model, tokenizer)
            target_demo_perplexity = get_perplexity_list(target_demo_df, model, tokenizer)

            islam_df['perplexity'] = islam_perplexity
            target_demo_df['perplexity'] = target_demo_perplexity

            print('Instances in demo 1 and 2: {}, {}'.format(len(islam_perplexity), len(target_demo_perplexity)))

            print('Mean and Std of unfiltered perplexities islam - Mean {}, Variance {}'.format(np.mean(islam_perplexity), np.std(islam_perplexity)))
            print('Mean and Std of unfiltered perplexities {} - Mean {}, Variance {}'.format(religion_list[index], np.mean(target_demo_perplexity), np.std(target_demo_perplexity)))

            assert len(islam_perplexity) == len(target_demo_perplexity)

            demo1_out = find_anomalies(np.array(islam_perplexity))
            demo2_out = find_anomalies(np.array(target_demo_perplexity))

            print(demo1_out, demo2_out)
            demo1_in = [d1 for d1 in islam_perplexity if d1 not in demo1_out]
            demo2_in = [d2 for d2 in target_demo_perplexity if d2 not in demo2_out]

            for i, (p1, p2) in enumerate(zip(islam_perplexity, target_demo_perplexity)):
                if p1 in demo1_out or p2 in demo2_out:
                    islam_df.drop(islam_df.loc[islam_df['perplexity'] == p1].index, inplace=True)
                    target_demo_df.drop(target_demo_df.loc[target_demo_df['perplexity'] == p2].index, inplace=True)

            print('Mean and Std of filtered perplexities islam - Mean {}, Variance {}'.format(np.mean(islam_df['perplexity']), np.std(islam_df['perplexity'])))
            print('Mean and Std of filtered perplexities {} - Mean {}, Variance {}'.format(religion_list[index], np.mean(target_demo_df['perplexity']), np.std(target_demo_df['perplexity'])))

            t_value, p_value = stats.ttest_ind(islam_perplexity, target_demo_perplexity, equal_var=False)

            print('Unfiltered perplexities - T value {} and P value {}'.format(t_value, p_value))
            #print(t_value, p_value)

            print("Length after outlier removal: ", len(islam_df['perplexity']), len(target_demo_df['perplexity']))
            t_unpaired, p_unpaired = stats.ttest_ind(islam_df['perplexity'].to_list(), target_demo_df['perplexity'].to_list(), equal_var=False)
            print('Student(unpaired) t-test, after outlier removal: t-value {}, p-value {}'.format(t_unpaired, p_unpaired))

            t_paired, p_paired = stats.ttest_rel(islam_df['perplexity'].to_list(), target_demo_df['perplexity'].to_list())
            print('Paired t-test, after outlier removal: t-value {}, p-value {}'.format(t_paired, p_paired))

'''
# ----------------------- Gender ------------------------------------------------

'''pretrained_gender_model = "../models/base_model/"

tokenizer = T5Tokenizer.from_pretrained(pretrained_gender_model)
model = T5ForConditionalGeneration.from_pretrained(pretrained_gender_model)


female_df = pd.read_csv('../data/gender_female_bias_manual_swapped_attr_test.txt', header=None)
male_df = pd.read_csv('../data/gender_male_bias_manual_swapped_attr_test.txt', header=None)

#print(female_df.head(n=10))
#print(male_df.head(n=10))
with open('../results/Base_FemaleVsMaleResults.txt', 'w') as file:
    sys.stdout = file

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
'''

# ----------------------------- Race ----------------------------------------

pretrained_model = "../models/base/model_files"

tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model)

race_list = ["black", "white", "native", "asian", "hispanic"]

for index in range(1, len(race_list)):
    black_df = pd.read_csv('../data/race_black_bias_manual_swapped_attr_test.txt', header=None)
    with open('../results/Base_BlackVs{}Results.txt'.format(race_list[index]), 'w') as file:
        sys.stdout = file
        target_demo_df = pd.read_csv('../data/race_{}_bias_manual_swapped_attr_test.txt'.format(race_list[index]),
                                 header=None)

        # print(black_df.head(n=10))
        # print(target_demo_df.head(n=10))

        black_perplexity = get_perplexity_list(black_df, model, tokenizer)
        target_demo_perplexity = get_perplexity_list(target_demo_df, model, tokenizer)

        black_df['perplexity'] = black_perplexity
        target_demo_df['perplexity'] = target_demo_perplexity

        print('Instances in demo 1 and 2: {}, {}'.format(len(black_perplexity), len(target_demo_perplexity)))

        print('Mean and Std of unfiltered perplexities black - Mean {}, Variance {}'.format(np.mean(black_perplexity),
                                                                                            np.std(black_perplexity)))
        print('Mean and Std of unfiltered perplexities {} - Mean {}, Variance {}'.format(race_list[index],
                                                                                         np.mean(target_demo_perplexity),
                                                                                         np.std(target_demo_perplexity)))

        assert len(black_perplexity) == len(target_demo_perplexity)

        demo1_out = find_anomalies(np.array(black_perplexity))
        demo2_out = find_anomalies(np.array(target_demo_perplexity))

        print(demo1_out, demo2_out)
        demo1_in = [d1 for d1 in black_perplexity if d1 not in demo1_out]
        demo2_in = [d2 for d2 in target_demo_perplexity if d2 not in demo2_out]

        for i, (p1, p2) in enumerate(zip(black_perplexity, target_demo_perplexity)):
            if p1 in demo1_out or p2 in demo2_out:
                black_df.drop(black_df.loc[black_df['perplexity'] == p1].index, inplace=True)
                target_demo_df.drop(target_demo_df.loc[target_demo_df['perplexity'] == p2].index, inplace=True)

        print(
            'Mean and Std of filtered perplexities black - Mean {}, Variance {}'.format(np.mean(black_df['perplexity']),
                                                                                        np.std(black_df['perplexity'])))
        print('Mean and Std of filtered perplexities {} - Mean {}, Variance {}'.format(race_list[index], np.mean(
            target_demo_df['perplexity']), np.std(target_demo_df['perplexity'])))

        t_value, p_value = stats.ttest_ind(black_perplexity, target_demo_perplexity, equal_var=False)

        print('Unfiltered perplexities - T value {} and P value {}'.format(t_value, p_value))
        # print(t_value, p_value)

        print("Length after outlier removal: ", len(black_df['perplexity']), len(target_demo_df['perplexity']))
        t_unpaired, p_unpaired = stats.ttest_ind(black_df['perplexity'].to_list(), target_demo_df['perplexity'].to_list(),
                                                 equal_var=False)
        print('Student(unpaired) t-test, after outlier removal: t-value {}, p-value {}'.format(t_unpaired, p_unpaired))

        t_paired, p_paired = stats.ttest_rel(black_df['perplexity'].to_list(), target_demo_df['perplexity'].to_list())
        print('Paired t-test, after outlier removal: t-value {}, p-value {}'.format(t_paired, p_paired))


# --------------------------- Language Model Perplexity --------------------

demo_list = ["religion", "races", "gender"]
base_model_path = "../models/base/model_files"
base_tokenizer = T5Tokenizer.from_pretrained(base_model_path)
base_model = T5ForConditionalGeneration.from_pretrained(base_model_path)

with open("../results/LMP.txt", 'w') as file:
    sys.stdout = file

    print("LM \t& Size \t& Regularized \t& Unregularized\\\\")

    for dm in demo_list:
        test_df = pd.read_csv('../data/{}_merged.txt'.format(dm), header=None)

        base_perplexity = get_model_perplexity(test_df, base_model, base_tokenizer)

        test_model_path = "../models/religion/model_files".format(dm)
        test_tokenizer = T5Tokenizer.from_pretrained(test_model_path)
        test_model = T5ForConditionalGeneration.from_pretrained(test_model_path)
        test_perplexity = get_model_perplexity(test_df, test_model, test_tokenizer)

        print('{} \t& {} \t& {} \t& {}\\\\'.format(dm, len(test_df), test_perplexity, base_perplexity))

