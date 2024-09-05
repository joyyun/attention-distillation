from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets import load_dataset

from bert_finetune import BERTLikeDataset

from torch.utils.data import DataLoader

import torch

import numpy as np

from tqdm import tqdm

from utils import get_nli_tok_func


def get_bert_logits(dset_name='cola'):
    '''

    utility function for other files to load cached BERT outputs

    '''
    # np_load_old = np.load
    # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    load_path_outputs = f"data/distill/BERTbase_{dset_name}_finetune-outputs.npz"
    bert_outputs = np.load(load_path_outputs)
    load_path_attns = f"data/distill/BERTbase_{dset_name}_finetune-attns.npz"
    bert_attns = np.load(load_path_attns)
    return bert_outputs, bert_attns


def main():

    # load data

    data_name = "cola"

    save_path_outputs = f"data/distill/BERTbase_{data_name}_finetune-outputs"
    save_path_attns = f"data/distill/BERTbase_{data_name}_finetune-attns"

    dset_splits = ['train', 'validation', 'test']

    n_classes = 2

    cola_raw = load_dataset("glue", "cola")

    # tokenize

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tok_func = get_nli_tok_func(tokenizer)

    cola_tok = cola_raw.map(tok_func, batched=True)

    bsz = 64

    device = 'cuda:0'

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)

    model.load_state_dict(torch.load('checkpoints/bert_finetune-2023_06_09-22_23.pth'))

    print('model finetune weights loaded')

    model.eval()

    model.to(device)

    bert_outputs_cache = {}
    bert_attns_cache = {}

    for split in dset_splits:

        dset = BERTLikeDataset(cola_tok[split])

        # create dataloaders (batches and randomizes samples)

        dl = DataLoader(dset, batch_size=bsz, shuffle=False, drop_last=False)

        model_outputs = []
        model_attentions = []

        with torch.no_grad():

            for batch in tqdm(dl, desc=split):

                (inp, am, _) = (t.to(device) for t in batch)

                out = model(input_ids=inp, attention_mask=am, return_dict=True, output_attentions=True)

                # (inp, tti, am, _) = (t.to(device) for t in batch)

                # out = model(input_ids=inp, attention_mask=am, token_type_ids=tti, return_dict=True)
                model_outputs.append(out.logits.cpu())
                model_attentions.append(out.attentions[-1].cpu())

        bert_outputs_cache[split] = torch.cat(model_outputs, dim=0).numpy()
        bert_attns_cache[split] = torch.cat(model_attentions, dim=0).numpy()

        assert len(bert_outputs_cache[split]) == cola_tok[split].num_rows
        assert len(bert_attns_cache[split]) == cola_tok[split].num_rows

    np.savez(save_path_outputs, **bert_outputs_cache)
    np.savez(save_path_attns, **bert_attns_cache)

    print(
        f"Outputs for splits {', '.join(dset_splits)} of dataset {data_name} saved to {save_path_outputs}")


if __name__ == '__main__':

    main()
