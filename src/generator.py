import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from tqdm import tqdm
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import pandas as pd

console = Console(record=True)

torch.manual_seed(0)  # pytorch random seed
np.random.seed(0)  # numpy random seed
torch.backends.cudnn.deterministic = True

class T5Generator():

    def __init__(self, model_params):
        self.model_params = model_params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_for_epoch(self, tokenizer, model, loader):
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for _, data in enumerate(loader, 0):
                y = data['target_ids'].to(self.device, dtype=torch.long)
                ids = data['source_ids'].to(self.device, dtype=torch.long)
                mask = data['source_mask'].to(self.device, dtype=torch.long)

                generated_ids = model.generate(input_ids=ids, attention_mask=mask, max_length=256, do_sample=True,
                                               num_return_sequences=1)
                preds = [tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=True) for g in
                         generated_ids]
                target = [tokenizer.decode(t, skip_special_tokens=False, clean_up_tokenization_spaces=True) for t in y]

                if _ % 1 == 0:
                    console.print(f'Completed {_}')

                predictions.extend(preds)
                actuals.extend(target)
        return predictions, actuals

    def generate(self, data_loader, output_file):
        console.log(f"[Loading Model]...\n")
        # Saving the model after training
        path = os.path.join(self.model_params["OUTPUT_PATH"], "model_files")
        model = T5ForConditionalGeneration.from_pretrained(path)
        tokenizer = T5Tokenizer.from_pretrained(path)
        model = model.to(self.device)
        # evaluating test dataset
        console.log(f"[Initiating Generation]...\n")
        for epoch in range(self.model_params["VAL_EPOCHS"]):
            predictions, actuals = self.generate_for_epoch(tokenizer, model, data_loader)
            final_df = pd.DataFrame(
                {'Generated Text': predictions, 'Actual Text': actuals})
            final_df.to_csv(os.path.join(self.model_params["OUTPUT_PATH"], output_file))

        console.save_text(os.path.join(self.model_params["OUTPUT_PATH"], 'logs.txt'))

        console.log(f"[Generation Completed.]\n")
        console.print(f"""[Model] Model saved @ {os.path.join(self.model_params["OUTPUT_PATH"], "model_files")}\n""")
        console.print(
            f"""[Validation] Generation on Validation data saved @ {os.path.join(self.model_params["OUTPUT_PATH"], output_file)}\n""")
        console.print(f"""[Logs] Logs saved @ {os.path.join(self.model_params["OUTPUT_PATH"], 'logs.txt')}\n""")



