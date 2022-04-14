from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
from trainer import DebiasingTrainer, LMHeadTrainer
from torch.utils.data import DataLoader
import data

BIAS_TYPE = data.RELIGION
# BIAS_TYPE = "NONE"

DEBIAS_MODEL = True  # Set to False to skip the debiasing process
LM_TRAINING = True  # Set to False to skip the LM training step

model_params_debias = {
    "OUTPUT_PATH": f"../models/{BIAS_TYPE}/exp_50/",
    "DOWNSTREAM_OUTPUT_PATH": f"../models/downstream/{BIAS_TYPE}/exp_50/",
    "MODEL": "bert-base-cased",  # model_type: t5-base/t5-large
    "TRAIN_EPOCHS": 30,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 32,  # max length of target text
    "EARLY_STOPPING_PATIENCE": 3,  # number of epochs before stopping training.
    "BATCH_SIZE": 64,  # Batch size to use
    "WORD_LIST": f"../word_lists/{BIAS_TYPE}.csv"
}

model_params_lm = {
    "OUTPUT_PATH": f"../models/{BIAS_TYPE}/exp_50/",
    "DOWNSTREAM_OUTPUT_PATH": f"../models/downstream/{BIAS_TYPE}/exp_50/",
    "MODEL": "bert-base-cased",  # model_type: t5-base/t5-large
    "LM_TRAIN_EPOCHS": 100,
    "LM_VAL_EPOCHS": 1,
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 32,  # max length of target text
    "EARLY_STOPPING_PATIENCE": 3,  # number of epochs before stopping training.
    "BATCH_SIZE": 64,  # Batch size to use
    "WORD_LIST": f"../word_lists/{BIAS_TYPE}.csv",
    "DOWNLOAD_CNN_DATA": False,  # Set to False to provide a path to the data
    "CNN_DATA_PATH": "../data/cnn_dailmail_saved.pkl",
    "MLM_PROBABILITY": 0.15,  # Masked LM probability
}

print(model_params_debias)
print(model_params_lm)
print(f"Debias Model: {DEBIAS_MODEL}\n")
print(f"LM Fine tune Model: {LM_TRAINING}\n")

# ==============================================================================
# ====                           DEBIASING STEPS                            ====
# ==============================================================================

tokenizer = BertTokenizer.from_pretrained(model_params_debias["MODEL"])

if DEBIAS_MODEL:

    print("Starting Model debiasing...")

    if BIAS_TYPE != data.MERGED:
        train, val, test = data.load_data_demographic(BIAS_TYPE)
    else:
        train, val, test = data.load_data_merged()

    if BIAS_TYPE == data.RELIGION:
        train = train[:5000]
        print("Taking subset of train dataset for Religion")

    train_dataset = data.SentencePairsDataset(
        train,
        tokenizer,
        model_params_debias["MAX_SOURCE_TEXT_LENGTH"],
    )
    val_dataset = data.SentencePairsDataset(
        val,
        tokenizer,
        model_params_debias["MAX_SOURCE_TEXT_LENGTH"],
    )
    test_dataset = data.SentencePairsDataset(
        test,
        tokenizer,
        model_params_debias["MAX_SOURCE_TEXT_LENGTH"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_params_debias["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=model_params_debias["BATCH_SIZE"], shuffle=True, num_workers=0
    )

    # Debiasing Training
    debiasing_trainer = DebiasingTrainer(model_params_debias, tokenizer)
    debiasing_trainer.train_model(train_dataloader, val_dataloader)

# ==============================================================================
# ====                          LM HEAD TRAINING                            ====
# ==============================================================================


if LM_TRAINING:

    print("Starting LM Training...")

    train, val, test = data.load_cnn_data(model_params_lm)
    ### Using 5% data (15000 training instances) like https://arxiv.org/pdf/2109.08565.pdf
    train = train[:15000]
    val = val[:6500]

    train_dataset = data.BertDataset(
        train,
        tokenizer,
        model_params_lm["MAX_SOURCE_TEXT_LENGTH"],
    )
    val_dataset = data.BertDataset(
        val,
        tokenizer,
        model_params_lm["MAX_SOURCE_TEXT_LENGTH"],
    )
    test_dataset = data.BertDataset(
        test,
        tokenizer,
        model_params_lm["MAX_SOURCE_TEXT_LENGTH"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=model_params_lm["MLM_PROBABILITY"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=model_params_lm["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=model_params_lm["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=model_params_lm["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )

    # Debiasing Training
    lm_trainer = LMHeadTrainer(model_params_lm, tokenizer)
    if DEBIAS_MODEL:
        lm_trainer.train_model(train_dataloader, val_dataloader, use_debiased_bert=True)
    else:
        lm_trainer.train_model(train_dataloader, val_dataloader, use_debiased_bert=False)
