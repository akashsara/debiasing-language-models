from transformers import BertTokenizer
from trainer import T5Trainer
from generator import T5Generator
from torch.utils.data import DataLoader
import data

BIAS_TYPE = "religion"

REGULARISATION_PARAM = 0.01

model_params = {
    "OUTPUT_PATH": f"../models/{BIAS_TYPE}/{REGULARISATION_PARAM}/",
    # "OUTPUT_PATH": "../models/",  # output path
    # "OUTPUT_PATH": "../models/{}/".format(BIAS_TYPE),  # output path
    "MODEL": "bert-base-cased",  # model_type: t5-base/t5-large
    "TRAIN_EPOCHS": 15,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 32,  # max length of target text
    "EARLY_STOPPING_PATIENCE": 3,  # number of epochs before stopping training.
    "SENTINEL_MASK_FRACTION": 0.15,  # Fraction of a sequence to sentinel mask
    "BATCH_SIZE": 32,  # Batch size to use
    "WORD_LIST": f"../word_lists/{BIAS_TYPE}.csv",
    "REGULARISATION_LAMBDA": REGULARISATION_PARAM,
}

# ==============================================================================
# ====                           DATA PREPARATION                           ====
# ==============================================================================

print(model_params)

tokenizer = BertTokenizer.from_pretrained(model_params["MODEL"])

train, val, test = data.load_data()

train_dataset = data.T5Dataset(
    train,
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
val_dataset = data.T5Dataset(
    val,
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
test_dataset = data.T5Dataset(
    test,
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
train_dataloader = DataLoader(
    train_dataset, batch_size=model_params["BATCH_SIZE"], shuffle=True, num_workers=0
)
val_dataloader = DataLoader(
    val_dataset, batch_size=model_params["BATCH_SIZE"], shuffle=True, num_workers=0
)
test_dataloader = DataLoader(
    test_dataset, batch_size=model_params["BATCH_SIZE"], shuffle=True, num_workers=0
)

# # ==============================================================================
# # ====                            MODELING STUFF                            ====
# # ==============================================================================

t5_trainer = T5Trainer(model_params, tokenizer)
t5_trainer.train_model(train_dataloader, val_dataloader)

# t5_generator = T5Generator(model_params)
# t5_generator.generate(test_dataloader, "predictions.csv")

"""
# ==============================================================================
# ====                            REDDITBIAS                                ====
# ==============================================================================

train, val, test = data.load_data_reddit()

train_dataset = data.T5Dataset(
    train,
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
val_dataset = data.T5Dataset(
    val,
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
test_dataset = data.T5Dataset(
    test,
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
train_dataloader = DataLoader(
    train_dataset, batch_size=model_params["BATCH_SIZE"], shuffle=True, num_workers=0
)
val_dataloader = DataLoader(
    val_dataset, batch_size=model_params["BATCH_SIZE"], shuffle=True, num_workers=0
)
test_dataloader = DataLoader(
    val_dataset, batch_size=model_params["BATCH_SIZE"], shuffle=True, num_workers=0
)


t5_trainer = T5Trainer(model_params, tokenizer)
t5_trainer.train_model(train_dataloader, val_dataloader)

t5_generator = T5Generator(model_params)
t5_generator.generate(test_dataloader, "predictions_reddit.csv")
"""
