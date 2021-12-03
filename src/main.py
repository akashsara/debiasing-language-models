from transformers import T5Tokenizer
from trainer import T5Trainer
from generator import T5Generator
from torch.utils.data import DataLoader
import data

BIAS_TYPE = 'races'

model_params = {
    "OUTPUT_PATH": "../models/{}/".format(BIAS_TYPE),  # output path
    "MODEL": "t5-base",  # model_type: t5-base/t5-large
    "TRAIN_EPOCHS": 5,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 32,  # max length of target text
    "EARLY_STOPPING_PATIENCE": 1,  # number of epochs before stopping training.
    "SENTINEL_MASK_FRACTION": 0.15,  # Fraction of a sequence to sentinel mask
    "BATCH_SIZE": 32,  # Batch size to use
    "WORD_LIST": "../data/{}.csv".format(BIAS_TYPE),
    "REGULARISATION_LAMBDA": 0.1
}

# ==============================================================================
# ====                           DATA PREPARATION                           ====
# ==============================================================================

print(model_params)

tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

train, val, test = data.load_data()

train_dataset = data.T5Dataset(
    train["article"][:5000],
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
val_dataset = data.T5Dataset(
    val["article"][:100],
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
test_dataset = data.T5Dataset(
    test["article"][:5000],
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

# ==============================================================================
# ====                            MODELING STUFF                            ====
# ==============================================================================

# t5_trainer = T5Trainer(model_params, tokenizer)
# t5_trainer.train_model(train_dataloader, val_dataloader)

t5_generator = T5Generator(model_params)
t5_generator.generate(test_dataloader, "predictions.csv")

'''
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
'''