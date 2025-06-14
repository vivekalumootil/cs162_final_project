from datasets import load_dataset, Dataset, concatenate_datasets
import torch
import torch.nn as nn
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaConfig
from transformers import RobertaModel, RobertaPreTrainedModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np
import os
import sys

if (len(sys.argv) != 2):
    print("Sorry, you haven't supplied the correct number of arguments")
    print("Please run: python3 submission.py [path to dataset]")
    print("Example: python3 submission.py ./arxiv_chatGPT.jsonl")
    exit(1)

'''
hc3_dataset = load_dataset(
    "Hello-SimpleAI/HC3",
    "all",
    cache_dir="./hc3_cache"
)
hc3_data = hc3_dataset["train"]
val_data = dataset["val"]
test_data = dataset["test"]
arxivGPT_dataset = load_dataset("json", data_files={"validation": "../data/arxiv_chatGPT.jsonl"})
arxivGPT_data = arxivGPT_dataset["validation"]
arxivCohere_dataset = load_dataset("json", data_files={"validation": "../data/arxiv_cohere.jsonl"})
arxivCohere_data = arxivCohere_dataset["validation"]
redditGPT_dataset = load_dataset("json", data_files={"validation": "../data/reddit_chatGPT.jsonl"})
redditGPT_data = redditGPT_dataset["validation"]
redditCohere_dataset = load_dataset("json", data_files={"validation": "../data/reddit_cohere.jsonl"})
redditCohere_data = redditCohere_dataset["validation"]

def process_hc3(split):
    processed = []
    for example in split:
        prompt = example["question"]
        for human_answer in example['human_answers']:
            processed.append({
                "text": human_answer,
                "label": 0
            })
        for ai_answer in example["chatgpt_answers"]:
            processed.append({
                "text": ai_answer,
                "label": 1
            })
    return Dataset.from_pandas(pd.DataFrame(processed))

'''
def process_M4(split):
    processed = []
    for example in split:
        prompt = example['prompt']
        processed.append({
            "text": example["human_text"],
            "label": 0  # Human
        })
        processed.append({
            "text": example["machine_text"],
            "label": 1  # AI
        })
    return Dataset.from_pandas(pd.DataFrame(processed))

'''
hc3_train_data = process_hc3(hc3_data)
arxivGPT_val_data = process_M4(arxivGPT_data)
arxivCohere_val_data = process_M4(arxivCohere_data)
redditGPT_val_data = process_M4(redditGPT_data)
redditCohere_val_data = process_M4(redditCohere_data)
'''

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

'''
hc3_train_data = hc3_train_data.map(tokenize, batched=True)
hc3_train_data = hc3_train_data.remove_columns(['text'])
hc3_train_data.set_format('torch')

arxivGPT_val_data = arxivGPT_val_data.map(tokenize, batched=True)
arxivGPT_val_data = arxivGPT_val_data.remove_columns(['text'])
arxivGPT_val_data.set_format('torch')

arxivCohere_val_data = arxivCohere_val_data.map(tokenize, batched=True)
arxivCohere_val_data = arxivCohere_val_data.remove_columns(['text'])
arxivCohere_val_data.set_format('torch')

redditGPT_val_data = redditGPT_val_data.map(tokenize, batched=True)
redditGPT_val_data = redditGPT_val_data.remove_columns(['text'])
redditGPT_val_data.set_format('torch')

redditCohere_val_data = redditCohere_val_data.map(tokenize, batched=True)
redditCohere_val_data = redditCohere_val_data.remove_columns(['text'])
redditCohere_val_data.set_format('torch')

val_data = concatenate_datasets([
    arxivGPT_val_data,
    arxivCohere_val_data,
    redditGPT_val_data,
    redditCohere_val_data
])
'''

input_file = sys.argv[1]
eval_dataset = load_dataset("json", data_files={"validation": input_file})["validation"]
print("---- Dataset Details ----")
print(eval_dataset)
eval_dataset = process_M4(eval_dataset)
eval_dataset = eval_dataset.map(tokenize, batched=True)
eval_dataset.set_format("torch")

class RobertaWithRegisterTokens(RobertaPreTrainedModel):
    def __init__(self, config, num_register_tokens=4):
        super().__init__(config)
        self.num_register_tokens = num_register_tokens
        
        # Base RoBERTa model 
        self.roberta = RobertaModel(config, add_pooling_layer = False)
        
        # Register tokens: learnable parameters [num_register_tokens, hidden_size]
        self.register_token = nn.Parameter(torch.randn(1, config.hidden_size))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch_size = input_ids.size(0)

        
        inputs_embeds = self.roberta.embeddings(input_ids)
        
        reg_tokens_expanded = self.register_token.unsqueeze(0).expand(batch_size, self.num_register_tokens, -1)
        
        extended_embeds = torch.cat([inputs_embeds, reg_tokens_expanded], dim=1)
        
        if attention_mask is not None:
            extra_attention = torch.ones(batch_size, self.num_register_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, extra_attention], dim=1)
        
        outputs = self.roberta(
            inputs_embeds=extended_embeds,
            attention_mask=attention_mask
        )
        
        # Use first token (<s>) hidden state for classification
        hidden_states = outputs.last_hidden_state  
        pooled_output = hidden_states[:, 0, :]     
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)   
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return (loss, logits) if loss is not None else logits

config = RobertaConfig.from_pretrained("roberta-base", num_labels=2)
model = RobertaWithRegisterTokens.from_pretrained("Roberta_4_noprompt_12", num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions

    probs = softmax(logits, axis=1)
    preds = np.argmax(probs, axis=1)

    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        roc_auc = float('nan')  # if all preds are one class

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": roc_auc,
    }

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

print("---- Inference Details----")
print(trainer.evaluate(eval_dataset=eval_dataset)) 
