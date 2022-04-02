from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
from trainer import DebiasingTrainer, LMHeadTrainer
from torch.utils.data import DataLoader
import data

BIAS_TYPE = data.RELIGION

REGULARISATION_PARAM = 0.01

model_params = {
    "OUTPUT_PATH": f"../models/{BIAS_TYPE}/{REGULARISATION_PARAM}/",
    "DOWNSTREAM_OUTPUT_PATH": f"../models/downstream/{BIAS_TYPE}/{REGULARISATION_PARAM}/",
    "MODEL": "bert-base-cased",  # model_type: t5-base/t5-large
    "TRAIN_EPOCHS": 30,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LM_TRAIN_EPOCHS": 30,
    "LM_VAL_EPOCHS": 1,
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 32,  # max length of target text
    "EARLY_STOPPING_PATIENCE": 3,  # number of epochs before stopping training.
    "SENTINEL_MASK_FRACTION": 0.15,  # Fraction of a sequence to sentinel mask
    "BATCH_SIZE": 64,  # Batch size to use
    "WORD_LIST": f"../word_lists/{BIAS_TYPE}.csv",
    "REGULARISATION_LAMBDA": REGULARISATION_PARAM,
    "DOWNLOAD_CNN_DATA": False,  # Set to False to provide a path to the data
    "CNN_DATA_PATH": "../data/cnn_dailmail_saved.pkl",
    "DEBIAS_MODEL": True,  # Set to False to skip the debiasing process
    "LM_TRAINING": True,  # Set to False to skip the LM training step
    "MLM_PROBABILITY": 0.15,  # Masked LM probability
}

print(model_params)
tokenizer = BertTokenizer.from_pretrained(model_params["MODEL"])

# ==============================================================================
# ====                           DEBIASING STEPS                            ====
# ==============================================================================

if model_params["DEBIAS_MODEL"]:

    print("Starting Model debiasing...")

    train, val, test = data.load_data(BIAS_TYPE)

    train_dataset = data.SentencePairsDataset(
        train,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
    )
    val_dataset = data.SentencePairsDataset(
        val,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
    )
    test_dataset = data.SentencePairsDataset(
        test,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_params["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=model_params["BATCH_SIZE"], shuffle=True, num_workers=0
    )

    # Debiasing Training
    debiasing_trainer = DebiasingTrainer(model_params, tokenizer)
    debiasing_trainer.train_model(train_dataloader, val_dataloader)

# ==============================================================================
# ====                          LM HEAD TRAINING                            ====
# ==============================================================================


if model_params["LM_TRAINING"]:

    print("Starting LM Training...")

    train, val, test = data.load_cnn_data(model_params)

    train_dataset = data.BertDataset(
        train,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
    )
    val_dataset = data.BertDataset(
        val,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
    )
    test_dataset = data.BertDataset(
        test,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=model_params["MLM_PROBABILITY"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=model_params["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=model_params["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=model_params["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )

    # Debiasing Training
    lm_trainer = LMHeadTrainer(model_params, tokenizer)
    lm_trainer.train_model(train_dataloader, val_dataloader)
