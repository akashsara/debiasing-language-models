import sys
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoModel
import os

console = Console(record=True)

torch.manual_seed(0)  # pytorch random seed
np.random.seed(0)  # numpy random seed
torch.backends.cudnn.deterministic = True


class DebiasingTrainer:
    def __init__(self, model_params, tokenizer, interleaving_epoch):
        self.model_params = model_params
        self.tokenizer = tokenizer
        self.interleaving_epoch = interleaving_epoch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Convert CSV -> Word Set
        # If multiple words are in a group for a class, we use the first word.

        df = pd.read_csv(model_params["WORD_LIST"])
        df.fillna("N/A", inplace=True)
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x.split(" _ ")[0])
            df[col] = df[col].apply(
                lambda x: tokenizer.encode(x, add_special_tokens=False)
            )
        self.word_set = list(df.T.to_dict(orient="list").values())
        self.num_classes = df.shape[1]
        param_tensor = torch.ones(self.num_classes) / self.num_classes
        torch.nn.init.uniform_(param_tensor)
        self.weighting_params = torch.nn.Parameter(
            param_tensor, requires_grad=True)

    def sentence_regularizer(
            self,
            sentence1: torch.Tensor,
            sentence2: torch.Tensor,
            word1: torch.Tensor,
            word2: torch.Tensor,
    ):
        # TODO: Do we want an absolute loss here?
        sentence_difference = sentence1 - sentence2
        word_difference = word1 - word2
        # TODO: What loss to use? MSE is the simple option
        return torch.nn.functional.mse_loss(sentence_difference, word_difference)

    def train(self, model, loader, optimizer):
        train_losses = []
        model.train()
        # print("Pre-Weighting parameters are: {}".format(str(self.weighting_params)))
        for _, data in tqdm(
                enumerate(loader, 0), total=len(loader), desc="Processing batches.."
        ):
            # Setup Data
            sentence1_y = data["sentence1_ids"].to(
                self.device, dtype=torch.long)
            sentence1_y_mask = data["sentence1_mask"].to(
                self.device, dtype=torch.long)
            sentence2_y = data["sentence2_ids"].to(
                self.device, dtype=torch.long)
            sentence2_y_mask = data["sentence2_mask"].to(
                self.device, dtype=torch.long)
            word1 = data["sensitive_word1_ids"].to(
                self.device, dtype=torch.float32)
            word2 = data["sensitive_word2_ids"].to(
                self.device, dtype=torch.float32)

            # Pass through model
            sentence1_y_hat = model(
                input_ids=sentence1_y, attention_mask=sentence1_y_mask
            )
            sentence2_y_hat = model(
                input_ids=sentence2_y, attention_mask=sentence2_y_mask
            )

            # TODO: How do we get sentence embeddings here?
            # We have individual word embeddings only.
            # Taking the mean right now but alternate methods are a possibility.
            # Also remember that 1 word = multiple tokens

            # Extracting the embeddings for each token
            # last_hidden_state: [batch_size, seq_len, hidden_state]
            sentence1_y_hat = sentence1_y_hat.last_hidden_state
            sentence2_y_hat = sentence2_y_hat.last_hidden_state

            # Extract embeddings for each sensitive word
            word1 = torch.matmul(word1, sentence1_y_hat)
            word2 = torch.matmul(word2, sentence2_y_hat)

            # Take Mean embedding of each word since different words
            # have a different number of tokens
            # [batch_size, hidden_state ]
            word1 = word1.mean(dim=1)
            word2 = word2.mean(dim=1)

            # Embedding of CLS token is the sentence embedding:
            # [batch_size, hidden_state]
            sentence1_y_hat = sentence1_y_hat[:, 0, :]
            sentence2_y_hat = sentence2_y_hat[:, 0, :]

            loss = self.sentence_regularizer(
                sentence1_y_hat, sentence2_y_hat, word1, word2
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        return train_losses

    def validate(self, model, loader):
        validate_losses = []
        model.eval()
        for _, data in tqdm(
                enumerate(loader, 0), total=len(loader), desc="Validating batches.."
        ):
            # Setup Data
            sentence1_y = data["sentence1_ids"].to(
                self.device, dtype=torch.long)
            sentence1_y_mask = data["sentence1_mask"].to(
                self.device, dtype=torch.long)
            sentence2_y = data["sentence2_ids"].to(
                self.device, dtype=torch.long)
            sentence2_y_mask = data["sentence2_mask"].to(
                self.device, dtype=torch.long)
            word1 = data["sensitive_word1_ids"].to(
                self.device, dtype=torch.float32)
            word2 = data["sensitive_word2_ids"].to(
                self.device, dtype=torch.float32)

            # Pass through model
            sentence1_y_hat = model(
                input_ids=sentence1_y, attention_mask=sentence1_y_mask
            )
            sentence2_y_hat = model(
                input_ids=sentence2_y, attention_mask=sentence2_y_mask
            )

            # TODO: How do we get sentence embeddings here?
            # We have individual word embeddings only.
            # Taking the mean right now but alternate methods are a possibility.
            # Also remember that 1 word = multiple tokens

            # Extracting the embeddings for each token
            # last_hidden_state: [batch_size, seq_len, hidden_state]
            sentence1_y_hat = sentence1_y_hat.last_hidden_state
            sentence2_y_hat = sentence2_y_hat.last_hidden_state

            # Extract embeddings for each sensitive word
            word1 = torch.matmul(word1, sentence1_y_hat)
            word2 = torch.matmul(word2, sentence2_y_hat)

            # Take Mean embedding of each word since different words
            # have a different number of tokens
            # [batch_size, hidden_state ]
            word1 = word1.mean(dim=1)
            word2 = word2.mean(dim=1)

            # Embedding of CLS token is the sentence embedding:
            # [batch_size, hidden_state]
            sentence1_y_hat = sentence1_y_hat[:, 0, :]
            sentence2_y_hat = sentence2_y_hat[:, 0, :]

            loss = self.sentence_regularizer(
                sentence1_y_hat, sentence2_y_hat, word1, word2
            )

            validate_losses.append(loss.item())
        return validate_losses

    def train_model(self, training_loader, validation_loader):
        console.log(f"""[Model]: Loading {self.model_params["MODEL"]}...\n""")
        if self.interleaving_epoch == 0:
            model = AutoModel.from_pretrained(self.model_params["MODEL"])
        else:
            print("Loading from previous interleaved epoch")
            model = AutoModel.from_pretrained(os.path.join(
                self.model_params["DOWNSTREAM_OUTPUT_PATH"], "model_bert_files"))
        model = model.to(self.device)
        parameters = [p for p in model.parameters()]
        optimizer = torch.optim.AdamW(
            params=parameters, lr=self.model_params["LEARNING_RATE"]
        )
        early_stopping = EarlyStopping(
            patience=self.model_params["EARLY_STOPPING_PATIENCE"],
            verbose=False,
            path=self.model_params["OUTPUT_PATH"],
        )
        # Training loop
        console.log(f"[Initiating Fine Tuning]...\n")
        for epoch in range(self.model_params["TRAIN_EPOCHS"]):
            console.log(
                f"[Epoch: {epoch + 1}/{self.model_params['TRAIN_EPOCHS']}]")
            train_losses = self.train(model, training_loader, optimizer)

            with torch.no_grad():
                valid_losses = self.validate(model, validation_loader)

            # print("TRAIN LOSS IS : " + str(train_losses))
            # print("VAL LOSS IS : " + str(valid_losses))

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            console.log(
                f"Train Loss: {train_loss} \n Val Loss: {valid_loss}\n\n")
            # early_stopping checks if the validation loss has decreased,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Saving the model after training
        path = os.path.join(self.model_params["OUTPUT_PATH"], "model_files")
        console.log(f"[Saving Model at {path}]...\n")
        model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class EarlyStopping:
    """
    Early stops training if the validation loss doesn't improve
    after a given patience threshold.
    """

    def __init__(
            self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int):
                How long to wait after last time validation loss improved.
                Default: 7
            verbose (bool):
                If True, prints a message for each validation loss improvement.
                Default: False
            delta (float):
                Minimum change in the monitored quantity to qualify as an
                improvement.
                Default: 0
            path (str):
                Path for the checkpoint to be saved to.
                Default: 'checkpoint.pt'
            trace_func (function):
                trace print function.
                Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = os.path.join(path, "best_model_checkpoint.pt")

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LMHeadTrainer:
    def __init__(self, model_params, tokenizer, interleaving_epoch):
        self.model_params = model_params
        self.tokenizer = tokenizer
        self.interleaving_epoch = interleaving_epoch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, model, loader, optimizer):
        train_losses = []
        model.train()
        for _, data in tqdm(
                enumerate(loader, 0), total=len(loader), desc="Processing batches.."
        ):
            # Setup Data
            input_ids = data["input_ids"].to(self.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(
                self.device, dtype=torch.long)
            labels = data["labels"].to(self.device, dtype=torch.long)

            # Pass through model
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        return train_losses

    def validate(self, model, loader):
        validate_losses = []
        model.eval()
        for _, data in tqdm(
                enumerate(loader, 0), total=len(loader), desc="Validating batches.."
        ):
            # Setup Data
            input_ids = data["input_ids"].to(self.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(
                self.device, dtype=torch.long)
            labels = data["labels"].to(self.device, dtype=torch.long)

            # Pass through model
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs["loss"]

            validate_losses.append(loss.item())
        return validate_losses

    def train_model(self, training_loader, validation_loader, use_debiased_bert=True):
        console.log(f"""[Model]: Loading {self.model_params["MODEL"]}...\n""")
        model = AutoModelForMaskedLM.from_pretrained(
            self.model_params["MODEL"])
        # Load debiased model
        if use_debiased_bert:
            print("Using debiased model...")
            model.base_model = AutoModel.from_pretrained(
                os.path.join(self.model_params["OUTPUT_PATH"], "model_files")
            )

        model = model.to(self.device)
        # Freeze layers
        # for param in model.bert.parameters():
        # param.requires_grad = False

        console.log(
            f"Total Parameters in Model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        console.log(
            f"Trainable Parameters in Model: {sum(p.numel() for p in model.parameters())}"
        )

        parameters = [p for p in model.parameters()]
        optimizer = torch.optim.AdamW(
            params=parameters, lr=self.model_params["LEARNING_RATE"]
        )
        early_stopping = EarlyStopping(
            patience=self.model_params["EARLY_STOPPING_PATIENCE"],
            verbose=False,
            path=self.model_params["OUTPUT_PATH"],
        )

        # Training loop
        console.log(f"[Initiating Fine Tuning]...\n")
        for epoch in range(self.model_params["LM_TRAIN_EPOCHS"]):
            console.log(
                f"[Epoch: {epoch + 1}/{self.model_params['LM_TRAIN_EPOCHS']}]")
            train_losses = self.train(model, training_loader, optimizer)

            with torch.no_grad():
                valid_losses = self.validate(model, validation_loader)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            console.log(
                f"Train Loss: {train_loss} \n Val Loss: {valid_loss}\n\n")
            # early_stopping checks if the validation loss has decreased,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Saving the model after training
        path = os.path.join(
            self.model_params["DOWNSTREAM_OUTPUT_PATH"], "model_files")
        path_bert = os.path.join(
            self.model_params["DOWNSTREAM_OUTPUT_PATH"], "model_bert_files")
        console.log(f"[Saving Model at {path}]...\n")
        model.save_pretrained(path)
        model.base_model.save_pretrained(path_bert)
        self.tokenizer.save_pretrained(path)
