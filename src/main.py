from transformers import T5Tokenizer
from trainer import T5Trainer
from torch.utils.data import DataLoader
import data

model_params = {
    "OUTPUT_PATH": "./models",  # output path
    "MODEL": "t5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 16,  # training batch size
    "VALID_BATCH_SIZE": 16,  # validation batch size
    "TRAIN_EPOCHS": 50,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
    "EARLY_STOPPING_PATIENCE": 1,  # number of epochs before stopping training.
    "SENTINEL_MASK_FRACTION": 0.15,  # Fraction of a sequence to sentinel mask
    "BATCH_SIZE": 16,  # Batch size to use
}

# ==============================================================================
# ====                           DATA PREPARATION                           ====
# ==============================================================================

tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

train, val, test = data.load_data()

train_dataset = data.T5Dataset(
    train[:100],
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
val_dataset = data.T5Dataset(
    val[:100],
    tokenizer,
    model_params["SENTINEL_MASK_FRACTION"],
    model_params["MAX_SOURCE_TEXT_LENGTH"],
    model_params["MAX_TARGET_TEXT_LENGTH"],
)
test_dataset = data.T5Dataset(
    test[:100],
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

t5_trainer = T5Trainer(model_params, tokenizer)
t5_trainer.train_model(train_dataloader, val_dataloader)
