from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Sequence
import transformers
import pickle
import pandas as pd
import random
from ast import literal_eval as make_tuple

RELIGION = "religion"
GENDER = "gender"
RACE = "race"
MERGED = "merged"

torch.manual_seed(0)  # pytorch random seed
np.random.seed(0)  # numpy random seed
torch.backends.cudnn.deterministic = True


def load_cnn_data(model_params) -> Tuple[Dict, Dict, Dict]:
    """
    Note: We can use a dict or something for easy swapping of datasets
    later on.
    """
    # https://huggingface.co/datasets/viewer/?dataset=cnn_dailymail
    if model_params["DOWNLOAD_CNN_DATA"]:
        dataset = load_dataset("cnn_dailymail", "3.0.0", keep_in_memory=True)
        with open("cnn_dailmail_saved.pkl", "wb") as filewrite:
            pickle.dump(dataset, filewrite)
    else:
        dataset = pickle.load(open(model_params["CNN_DATA_PATH"], "rb"))
    train, val, test = dataset["train"], dataset["validation"], dataset["test"]
    return train["article"], val["article"], test["article"]


def get_samples(array, samples_needed):
    samples = []
    while len(samples) < samples_needed:
        i = np.random.randint(0, len(array), 1)
        array_i = array[i][0]
        array_i = [x for x in array_i if len(make_tuple(x)[1]) > 0]
        if len(array_i) < 2:
            continue
        array_i_1, array_i_2 = np.random.choice(array_i, size=2, replace=False)
        array_i_1 = array_i_1.replace("\\n", "")
        array_i_2 = array_i_2.replace("\\n", "")
        record = [make_tuple(array_i_1), make_tuple(array_i_2)]
        samples.append(record)
    return samples


def get_pairs(array, suppress_attribute_words=[]):
    dataset = []
    for i in range(len(array)):
        array_i = array[i]
        array_i = [x for x in array_i if x and len(make_tuple(x)[1]) > 0]
        if len(array_i) < 2:
            continue
        for j in range(len(array_i)):
            for k in range(j+1, len(array_i)):
                array_i_1 = array_i[j]
                array_i_2 = array_i[k]
                # Supress some sensitive word entries just for experimentation
                if make_tuple(array_i_1)[1] not in suppress_attribute_words and make_tuple(array_i_2)[1] not in suppress_attribute_words:
                    record = (make_tuple(array_i_1), make_tuple(array_i_2))
                    dataset.append(record)
    deduped_dataset = list(set(dataset))
    deduped_dataset = [list(d) for d in deduped_dataset]
    return deduped_dataset


def load_data_demographic(bias_type, suppress_attribute_words=[]) -> Tuple[Dict, Dict, Dict]:
    if bias_type == RELIGION:
        val_size = 1000
        csv_file = "../data/column_based_religion_data.csv"
    elif bias_type == GENDER:
        val_size = 200
        csv_file = "../data/column_based_gender_data.csv"
    elif bias_type == RACE:
        val_size = 1000
        csv_file = "../data/column_based_race_data.csv"

    df = pd.read_csv(csv_file, header=None, keep_default_na=False)
    val_size = 1000
    pairs = get_pairs(
        df.to_numpy(), suppress_attribute_words=suppress_attribute_words)
    train = pairs[:-val_size]
    val = pairs[-val_size:]
    return train, val, {}


# def load_data_merged() -> Tuple[Dict, Dict, Dict]:
#     religion_data_train, religion_data_val, _ = load_data_demographic(RELIGION)
#     gender_data_train, gender_data_val, _ = load_data_demographic(GENDER)
#     race_data_train, race_data_val, _ = load_data_demographic(RACE)
#
#     random.shuffle(religion_data_train)
#     random.shuffle(gender_data_train)
#     random.shuffle(race_data_train)
#     random.shuffle(religion_data_val)
#     random.shuffle(gender_data_val)
#     random.shuffle(race_data_val)
#
#     train = religion_data_train[:1000] + gender_data_train[:1000] + race_data_train[:1000]
#     val = religion_data_val[:200] + gender_data_val[:200] + race_data_val[:200]
#
#     return train, val, {}


def load_data_reddit() -> Tuple[Dict, Dict, Dict]:
    """
    Loading texts from RedditBias Data
    """
    # demographics = ["gender", "race", "religion1", "religion2"]
    demographics = ["religion1", "religion2"]
    train, val, test = [], [], []

    train_text = None
    valid_text = None
    test_text = None

    for d in demographics:
        with open("../data/" + str(d) + "_bias_manual_train.txt") as file:
            train_text = file.readlines()
            file.close()

        with open("../data/" + str(d) + "_bias_manual_valid.txt") as file:
            valid_text = file.readlines()
            file.close()

        with open("../data/" + str(d) + "_bias_manual_swapped_attr_test.txt") as file:
            test_text = file.readlines()
            file.close()

        train += train_text
        val += valid_text
        test += test_text

    return train, val, test


class SentencePairsDataset(Dataset):
    def __init__(
            self,
            data: List[str],
            tokenizer: transformers.PreTrainedTokenizer,
            max_source_length: int,
    ):
        """
        Args:
            data: A list of articles
            tokenizer: Tokenizer to use
            max_source_length: maximum source sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get item from dataset
        pair1, pair2 = self.data[idx]
        sentence1 = pair1[0]
        sensitive_word1 = pair1[1][0]
        sentence2 = pair2[0]
        sensitive_word2 = pair2[1][0]

        # Tokenize, Encode, Pad/Truncate & Get Attention Mask
        sentence1_encoding = self.tokenizer(
            sentence1,
            padding="max_length",
            max_length=self.max_source_length,
            truncation=True,
            is_split_into_words=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        sentence2_encoding = self.tokenizer(
            sentence2,
            padding="max_length",
            max_length=self.max_source_length,
            truncation=True,
            is_split_into_words=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        sensitive_word1_encoding = self.tokenizer(
            sensitive_word1,
            is_split_into_words=True,
            return_attention_mask=False,
            return_tensors="pt",
        )
        sensitive_word2_encoding = self.tokenizer(
            sensitive_word2,
            is_split_into_words=True,
            return_attention_mask=False,
            return_tensors="pt",
        )

        sentence1_ids = sentence1_encoding.input_ids[0]
        sentence2_ids = sentence2_encoding.input_ids[0]
        sentence1_mask = sentence1_encoding.attention_mask[0]
        sentence2_mask = sentence2_encoding.attention_mask[0]
        # Ignore CLS & SEP when getting sensitive word ids
        sensitive_word1_ids = sensitive_word1_encoding.input_ids[0][1:-1]
        sensitive_word2_ids = sensitive_word2_encoding.input_ids[0][1:-1]
        sensitive_word1_mask = self.mask_sensitive_word(
            sentence1_ids, sensitive_word1_ids
        )
        sensitive_word2_mask = self.mask_sensitive_word(
            sentence2_ids, sensitive_word2_ids
        )

        # Return
        return {
            "sentence1_ids": sentence1_ids,
            "sentence2_ids": sentence2_ids,
            "sentence1_mask": sentence1_mask,
            "sentence2_mask": sentence2_mask,
            "sensitive_word1_ids": sensitive_word1_mask,
            "sensitive_word2_ids": sensitive_word2_mask,
        }

    def mask_sensitive_word(self, sentence_ids, sensitive_word_ids):
        sensitive_word_mask = torch.zeros_like(sentence_ids)
        word_len = len(sensitive_word_ids)
        for i in range(len(sentence_ids)):
            if torch.equal(sentence_ids[i: i + word_len], sensitive_word_ids):
                sensitive_word_mask[i: i + word_len] = 1
        return sensitive_word_mask


class BertDataset(Dataset):
    def __init__(
            self,
            data: List[str],
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int,
    ):
        """
        Args:
            data: A list of articles
            tokenizer: Tokenizer to use
            mask_fraction: Percentage of the sequence to sentinel mask
            max_source_length: maximum source sequence length
            max_target_length: maximum target sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get item from dataset
        sequence = self.data[idx]
        # Tokenize, Encode, Pad/Truncate & Get Attention Mask
        encoding = self.tokenizer(
            sequence,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            is_split_into_words=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # Return
        return {
            "input_ids": encoding.input_ids[0],
            "attention_mask": encoding.attention_mask[0],
        }


class T5Dataset(Dataset):
    def __init__(
            self,
            data: List[str],
            tokenizer: transformers.PreTrainedTokenizer,
            mask_fraction: int,
            max_source_length: int,
            max_target_length: int,
    ):
        """
        Args:
            data: A list of articles
            tokenizer: Tokenizer to use
            mask_fraction: Percentage of the sequence to sentinel mask
            max_source_length: maximum source sequence length
            max_target_length: maximum target sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.mask_fraction = mask_fraction
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get item from dataset
        sequence = self.data[idx].split()
        # Get indices of words to mask
        mask_indices = self.get_mask_ids(len(sequence))
        # Apply sentinel masking
        masked_inputs, masked_target = self.add_sentinel_tokens(
            sequence, mask_indices)
        # Tokenize, Encode, Pad/Truncate & Get Attention Mask
        encoding = self.tokenizer(
            masked_inputs,
            padding="max_length",
            max_length=self.max_source_length,
            truncation=True,
            is_split_into_words=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            masked_target,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            is_split_into_words=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # Return
        return {
            "source_ids": encoding.input_ids[0],
            "source_mask": encoding.attention_mask[0],
            "target_ids": labels.input_ids[0],
            "target_mask": labels.attention_mask[0],
        }

    def get_mask_ids(self, sequence_length: int) -> Sequence[int]:
        """
        Pick self.mask_fraction tokens to mask.
        """
        num_mask_tokens = int(np.round(sequence_length * self.mask_fraction))
        mask_indices = []
        while len(mask_indices) < num_mask_tokens:
            index = np.random.randint(0, sequence_length)
            if index not in mask_indices:
                mask_indices.append(index)
        return mask_indices

    def add_sentinel_tokens(
            self, sequence: List[str], mask_indices: List[int]
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Apply sentinel masking
        Ref: https://arxiv.org/pdf/1910.10683.pdf (Figure 2)
        Note: This can probably be improved.
        """
        masked_input, masked_target = [], []
        input_sentinels = -1
        target_sentinels = -1
        previous_masked_input = False
        previous_masked_target = False
        for i, word in enumerate(sequence):
            if i in mask_indices:
                previous_masked_target = False
                if not previous_masked_input:
                    input_sentinels += 1
                    masked_input.append(f"<extra_id_{input_sentinels}>")
                masked_target.append(word)
                previous_masked_input = True
            else:
                previous_masked_input = False
                if not previous_masked_target:
                    target_sentinels += 1
                    masked_target.append(f"<extra_id_{target_sentinels}>")
                masked_input.append(word)
                previous_masked_target = True
        return masked_input, masked_target
