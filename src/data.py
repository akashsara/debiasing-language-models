from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Sequence
import transformers
import pickle
import pandas as pd
from ast import literal_eval as make_tuple

torch.manual_seed(0)  # pytorch random seed
np.random.seed(0)  # numpy random seed
torch.backends.cudnn.deterministic = True


def get_samples(array, samples_needed):
    samples = []
    while len(samples) < samples_needed:
        if len(samples) == 95:
            print("Here")
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


def load_data() -> Tuple[Dict, Dict, Dict]:
    df = pd.read_csv('../data/column_based_religion_data.csv')
    train_array, val_array, test_array = df[:-100].to_numpy(), df[-100:-90].to_numpy(), df[-90:].to_numpy()

    train = get_samples(train_array, 100)
    val = get_samples(val_array, 10)
    test = get_samples(test_array, 10)

    return train, val, test


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
        sensitive_word1_ids = sensitive_word1_encoding.input_ids[0][:-1]
        sensitive_word2_ids = sensitive_word2_encoding.input_ids[0][:-1]
        sensitive_word1_mask = self.mask_sensitive_word(sentence1_ids, sensitive_word1_ids)
        sensitive_word2_mask = self.mask_sensitive_word(sentence2_ids, sensitive_word2_ids)

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
            if torch.equal(sentence_ids[i:i+word_len], sensitive_word_ids):
                sensitive_word_mask[i:i+word_len] = 1
        return sensitive_word_mask

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
