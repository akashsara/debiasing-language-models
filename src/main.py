from transformers import T5Tokenizer
from trainer import T5Trainer

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
        "early_stopping_patience": 1,  # number of epochs before stopping training.
}

tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
t5_trainer = T5Trainer(model_params, tokenizer)
t5_trainer.train_model()