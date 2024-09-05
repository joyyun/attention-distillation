import numpy as np
import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
model.load_state_dict(torch.load(
    'checkpoints/distilbert_distfinetune_attention_bert-2023_06_09-22_55.pth'))  # change path

class_names = ['unacceptable', 'acceptable']

print("running lime on distilbert attention distillation...")


def predictor(texts):
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    tensor_logits = outputs[0]
    probas = F.softmax(tensor_logits).detach().numpy()
    return probas


text1 = 'She talked to John or Mary but I don\'t know which.'  # true label is 1 easy
# true label is 0 easy
text2 = 'I know that Meg\'s attracted to Harry, but they don\'t know who.'
text3 = 'Everyone relies on someone. It\'s unclear who.'  # true label 1 hard
text4 = 'I demand that the more John eat, the more he pays.'  # true label 0 hard

print(tokenizer(text4, return_tensors='pt', padding=True))


explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(
    text4, predictor, num_features=5, num_samples=200)
exp.save_to_file('./lime_test_distilbert_attention_distill4.html')
# exp.show_in_notebook(text=text)
