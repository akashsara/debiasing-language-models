"""
This script performs Student t-test on the perplexity distribution of two sentences groups with contrasting targets
"""
import pandas as pd
import numpy as np
from scipy import stats
import helper_functions as helpers
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelWithLMAndDebiasHead
import time
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import numpy as np
from outliers import smirnov_grubbs as grubbs


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
    for idx, row in df.iterrows():
        try:
            perplexity = helpers.perplexity_score(row['comments_processed'], m, t)
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
                perplexity = helpers.perplexity_score(row['comments_1'], m, t)
            else:
                perplexity = helpers.perplexity_score(row['comments_2'], m, t)
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
    model_perplexity = helpers.model_perplexity(df['comments_processed'], m, t)
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


start = time.time()

data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
exp_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Experiments/execution_logs/'

ON_SET = True
GET_PERPLEXITY = True
ON_TESTSET = False
GET_PERPLEXITY_TEST = False
REDUCE_SET = False # set to true to save dataset with outliers removed

demo = 'religion1' # 'orientation' # 'gender' # 'religion2' # 'race' # 'religion2
demo_1 = 'jews' # 'lgbtq' # 'female' # 'muslims' # 'black' # 'black_pos'
demo_2 = 'christians' # 'straight' # 'male' # 'white' # 'white_pos'  # 'white' #'christians2'
input_file_suffix = '_biased_test_reduced' #'_processed_phrase_biased_testset_reduced' #
output_file_suffix = '_perplex_phrase_biased' # '_perplex'

debiasing_head = 'EqualisingLoss' # 'CosineDist'
pretrained_model = 'microsoft/DialoGPT-small' # 'bert_base_uncased' # 'gpt2'
    # '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/models/religion1/eq_loss_0/'
    # 'microsoft/DialoGPT-small'
    # '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/models/religion2/lm_loss_swapped_targets/'

if ON_SET:
    logging.basicConfig(filename=exp_path+'measure_bias'+demo+'.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
else:
    logging.basicConfig(filename=exp_path+'measure_bias'+demo+'_test.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

pd.set_option('max_colwidth', 600)
pd.options.display.max_columns = 10

if GET_PERPLEXITY:

    print('Calculating perplexity for demo: {}'.format(demo))
    race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_suffix + '.csv')
    race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + input_file_suffix + '.csv')

     # 'gpt2' # 'roberta-base' # 'bert-base-uncased' #  #'ctrl'
    # "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'minimaxir/reddit' # 'xlnet-large-cased'
    # pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/colab_outputs/religion1/eq_loss/'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    # model = AutoModelWithLMHead.from_pretrained(pretrained_model)

    # model = AutoModelWithLMAndDebiasHead.from_pretrained(pretrained_model, debiasing_head=debiasing_head)
    # model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)

    # for n, p in model.named_parameters():
    #     if 'wte.weight' in n:
    #         print('model params {}, {}'.format(n, p))
    #         print(p.shape)

    race_1_perplexity = get_perplexity_list(race_df, model, tokenizer)
    print('Done with demo1 perplexity in {} on set'.format((time.time() - start)/60))
    race_2_perplexity = get_perplexity_list(race_df_2, model, tokenizer)

    # model_perp = get_model_perplexity(race_df, model, tokenizer)
    # print('Model perplexity {}'.format(model_perp))

    logging.info('Time to get perplexity scores {}'.format((time.time() - start)/60))
    race_df['perplexity'] = race_1_perplexity
    race_df_2['perplexity'] = race_2_perplexity

    # race_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_file_suffix + '.csv')
    # race_df_2.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + output_file_suffix +'.csv')
else:
    logging.info('Getting saved perplexity')
    print('Getting saved perplexity')
    race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_file_suffix +'.csv')
    race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + output_file_suffix +'.csv')
    race_1_perplexity = race_df['perplexity']
    race_2_perplexity = race_df_2['perplexity']


logging.debug('Instances in demo 1 and 2: {}, {}'.format(len(race_1_perplexity), len(race_2_perplexity)))
logging.debug('Mean and variance of unfiltered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_perplexity), np.var(race_1_perplexity)))
logging.debug('Mean and variance of unfiltered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_perplexity), np.var(race_2_perplexity)))

print('Mean and std of unfiltered perplexities demo1 - Mean {}, Std {}'.format(np.mean(race_1_perplexity), np.std(race_1_perplexity)))
print('Mean and std of unfiltered perplexities demo2 - Mean {}, Std {}'.format(np.mean(race_2_perplexity), np.std(race_2_perplexity)))


print(len(race_1_perplexity), len(race_2_perplexity))

demo1_out = find_anomalies(np.array(race_1_perplexity))
demo2_out = find_anomalies(np.array(race_2_perplexity))

print(demo1_out, demo2_out)
demo1_in = [d1 for d1 in race_1_perplexity if d1 not in demo1_out]
demo2_in = [d2 for d2 in race_2_perplexity if d2 not in demo2_out]

for i, (p1, p2) in enumerate(zip(race_1_perplexity, race_2_perplexity)):
    if p1 in demo1_out or p2 in demo2_out:
        print('Outlier in demo1 is {}'.format(race_df.loc[race_df['perplexity'] == p1]))
        print('Outlier in demo2 is {}'.format(race_df_2.loc[race_df_2['perplexity'] == p2]))
        race_df.drop(race_df.loc[race_df['perplexity'] == p1].index, inplace=True)
        race_df_2.drop(race_df_2.loc[race_df_2['perplexity'] == p2].index, inplace=True)

if REDUCE_SET:
    print('DF shape after reducing {}'.format(race_df.shape))
    print('DF 2 shape after reducing {}'.format(race_df_2.shape))
    race_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_suffix + '_reduced.csv', index=False)
    race_df_2.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + input_file_suffix + '_reduced.csv', index=False)

    print(len(race_df['perplexity']), len(race_df_2['perplexity']))
    print('Mean and std of filtered perplexities demo1 - Mean {}, Std {}'.format(np.mean(race_df['perplexity']),
                                                                                 np.std(race_df['perplexity'])))
    print('Mean and std of filtered perplexities demo2 - Mean {}, Std {}'.format(np.mean(race_df_2['perplexity']),
                                                                                 np.std(race_df_2['perplexity'])))

    t_unpaired, p_unpaired = stats.ttest_ind(race_df['perplexity'].to_list(), race_df_2['perplexity'].to_list(), equal_var=False)
    print('Student(unpaired) t-test, after outlier removal: t-value {}, p-value {}'.format(t_unpaired, p_unpaired))

    t_paired, p_paired = stats.ttest_rel(race_df['perplexity'].to_list(), race_df_2['perplexity'].to_list())
    print('Paired t-test, after outlier removal: t-value {}, p-value {}'.format(t_paired, p_paired))

t_value, p_value = stats.ttest_rel(race_1_perplexity, race_2_perplexity)

print('Mean and std of unfiltered perplexities demo1 - Mean {}, Std {}'.format(np.mean(race_1_perplexity),
                                                                               np.std(race_1_perplexity)))
print('Mean and std of unfiltered perplexities demo2 - Mean {}, Std {}'.format(np.mean(race_2_perplexity),
                                                                               np.std(race_2_perplexity)))
print('Unfiltered perplexities - T value {} and P value {}'.format(t_value, p_value))
print(t_value, p_value)

logging.info('Total time taken {}'.format((time.time() - start)/60))
