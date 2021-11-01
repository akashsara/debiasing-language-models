import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from tqdm import tqdm
import numpy as np
from transformers import T5ForConditionalGeneration
import os

console = Console(record=True)

torch.manual_seed(0)  # pytorch random seed
np.random.seed(0)  # numpy random seed
torch.backends.cudnn.deterministic = True


class T5Trainer:
    def __init__(self, model_params, tokenizer):
        self.model_params = model_params
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def regulariser(self, model):
        return 0

    def train(self, model, loader, optimizer):
        train_losses = []
        model.train()
        for _, data in tqdm(
            enumerate(loader, 0), total=len(loader), desc="Processing batches.."
        ):
            y = data["target_ids"].to(self.device, dtype=torch.long)
            lm_labels = y.clone()
            lm_labels[y == self.tokenizer.pad_token_id] = -100
            ids = data["source_ids"].to(self.device, dtype=torch.long)
            mask = data["source_mask"].to(self.device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
            loss = outputs[0] + self.regulariser(model)

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
            y = data["target_ids"].to(self.device, dtype=torch.long)
            lm_labels = y.clone()
            lm_labels[y == self.tokenizer.pad_token_id] = -100
            ids = data["source_ids"].to(self.device, dtype=torch.long)
            mask = data["source_mask"].to(self.device, dtype=torch.long)
            outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
            loss = outputs[0]
            validate_losses.append(loss.item())
        return validate_losses

    def train_model(self, training_loader, validation_loader):
        console.log(f"""[Model]: Loading {self.model_params["MODEL"]}...\n""")
        model = T5ForConditionalGeneration.from_pretrained(self.model_params["MODEL"])
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=self.model_params["LEARNING_RATE"]
        )
        early_stopping = EarlyStopping(
            patience=self.model_params["EARLY_STOPPING_PATIENCE"],
            verbose=False,
            path=f'{self.model_params["OUTPUT_PATH"]}/best_model_checkpoint.pt',
        )
        # Training loop
        console.log(f"[Initiating Fine Tuning]...\n")
        for epoch in range(self.model_params["TRAIN_EPOCHS"]):
            train_losses = self.train(model, training_loader, optimizer)
            valid_losses = self.validate(model, validation_loader)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        console.log(f"[Saving Model]...\n")
        # Saving the model after training
        path = os.path.join(self.model_params["OUTPUT_PATH"], "model_files")
        model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                Default: 0
            path (str): Path for the checkpoint to be saved to.
                Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

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
