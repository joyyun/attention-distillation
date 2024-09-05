from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score

model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) # Note: randomness for pretrained weights
model1.load_state_dict(torch.load('checkpoints/bert_finetune-2023_06_09-01_15.pth'))
tokenizer1 = AutoTokenizer.from_pretrained("bert-base-uncased")

model2 = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model2.load_state_dict(torch.load('checkpoints/distilbert_finetune-2023_06_09-02_00.pth'))
# tokenizer2 = AutoTokenizer.from_pretrained("distilbert-base-uncased")

cola_raw = load_dataset("glue", "cola")

inputs = tokenizer1(cola_raw['validation']['sentence'], truncation=True, padding=True)

batch_size = 32

data_loader1 = DataLoader(list(zip(torch.LongTensor(inputs['input_ids']), inputs['attention_mask'], cola_raw['validation']['label'])),
                         batch_size=batch_size,
                         shuffle=False)

data_loader2 = DataLoader(list(zip(torch.LongTensor(inputs['input_ids']), inputs['attention_mask'], cola_raw['validation']['label'])),
                         batch_size=batch_size,
                         shuffle=False)

model1.eval()
predictions1 = []
with torch.no_grad():
    for input_ids, attention_mask, labels in data_loader1:
        input_ids = torch.LongTensor(input_ids)
        input_ids = torch.transpose(input_ids, 0, 1)
        # print(torch.transpose(input_ids, 0, 1).shape)
        # print(torch.LongTensor([tensor.tolist() for tensor in attention_mask]).shape)
        outputs = model1(torch.LongTensor(input_ids), attention_mask=torch.LongTensor([tensor.tolist() for tensor in attention_mask]))
        logits = outputs.logits
        predicted_labels = logits.argmax(dim=1)
        predictions1.extend(predicted_labels.tolist())

model2.eval()
predictions2 = []
with torch.no_grad():
    for input_ids, attention_mask, labels in data_loader2:
        input_ids = torch.LongTensor(input_ids)
        input_ids = torch.transpose(input_ids, 0, 1)
        outputs = model2(torch.LongTensor(input_ids), attention_mask=torch.LongTensor([tensor.tolist() for tensor in attention_mask]))
        logits = outputs.logits
        predicted_labels = logits.argmax(dim=1)
        predictions2.extend(predicted_labels.tolist())

print("Model 1 predicts this many 1's: " + str(predictions1.count(1)))
print("Model 2 predicts this many 1's: " + str(predictions2.count(1)))

score = cohen_kappa_score(predictions1, predictions2)
print("Cohen Kappa score is: " + str(score))
