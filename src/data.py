from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple
import transformers


def load_data() -> Tuple[Dict, Dict, Dict]:
    """
    Note: We can use a dict or something for easy swapping of datasets
    later on.
    """
    # https://huggingface.co/datasets/viewer/?dataset=cnn_dailymail
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train, val, test = dataset["train"], dataset["validation"], dataset["test"]
    return train, val, test


class T5Dataset(Dataset):
    def __init__(
        self,
        data: list[str],
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
            sequence, mask_indices
        )
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
            return_attention_mask=False,
            return_tensors="pt",
        )
        # Return
        return {
            "source_ids": encoding.input_ids[0],
            "source_mask": encoding.attention_mask[0],
            "target_ids": labels.input_ids[0],
        }

    def get_mask_ids(self, sequence_length: int) -> list[int]:
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
        self, sequence: list[str], mask_indices: list[int]
    ) -> Tuple[list[str], list[str]]:
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
