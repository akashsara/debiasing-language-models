# Debiasing Multiclass Demographics in Language Models

Akash Saravanan, Dhruv Mullick, Habibur Rahman

---
### Abstract:

The majority of current approaches towards debiasing language models consider only a binary demographic such as male vs female or Caucasian vs African-American. While these methods are effective, their target is not representative of the real world. We propose a novel method to debias multiclass demographics. Our technique involves modification of the loss function to add a regularization term. This method can be scaled to any number of classes. We further compare our debiasing method with a base model to display its effectiveness. Finally, we contribute several word lists for the different demographics we consider. These word lists, meant for use in debiasing tasks, are compiled from a combination of prior work and online sources.

### Usage Instructions:

1. Create a new environment and install the required dependencies from `requirements.txt`.
2. Train your model using `src/main.py`. Enter the configuration details that you wish to use there. 
   * At present the code only supports T5 but changing to a different model is simple - just import the relevant model from transformers and change the preprocessing function.
   * Our word lists are available in the `word_lists` directory for ease-of-access. 
3. Run the evaluation script `evaluator.py` to calculate the BLEU, ROUGE and METEOR scores. 
4. Run the evaluation script `measure_bias_attribute_swap.py` to calculate the Language Model Perplexity (LMP) and Language Model Bias (LMB). Note that you may have to uncomment some parts of this code.

### Notes on dataset
CNN Dataset is pickled and should be downloaded from this [link](https://drive.google.com/file/d/1NM7Vev00Cxw2xlDt9zU8P-JgELMCDcm0/view?usp=sharing) into the src folder.

### Acknowledgements

We use the evaluation functions from https://github.com/umanlp/RedditBias with some modifications for Language Model Bias and Language Model Perplexity.