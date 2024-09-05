from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# need to make this
from utils import log_metrics, current_pst_time, acc_func, get_nli_tok_func
from tqdm import tqdm, trange

# ==== BIG CODE BLOCKS ====


class BERTLikeDataset(Dataset):
    def __init__(self, tok_dset):
        self.idx = tok_dset['idx']
        self.sentence = tok_dset['sentence']
        self.labels = tok_dset['label']
        self.input_ids = tok_dset['input_ids']
        self.attention_mask = tok_dset['attention_mask']
        #print("input ids: ", self.input_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sentence = torch.LongTensor(self.input_ids[idx])
        ams = torch.FloatTensor(self.attention_mask[idx])
        #sentence = torch.LongTensor(self.sentence[idx])
        label = self.labels[idx]
        return sentence, ams, label


def bert_epoch_loop(model, dl, loss_func, device='cpu', optimizer=None):
    # set training or validation mode
    do_train = optimizer is not None
    model.train() if do_train else model.eval()

    epoch_preds = []
    labels = []
    accum_loss = 0
    for i, batch in enumerate(tqdm(dl)):
        (inp, _, lbl) = (t.to(device) for t in batch)
        out = model(input_ids=inp, return_dict=True)
        loss = loss_func(out.logits, lbl)
        if do_train:
            optimizer.zero_grad()  # at every batch
            loss.backward()
            optimizer.step()
        lbl_pred = torch.argmax(out.logits.detach(), dim=1)
        epoch_preds += lbl_pred.cpu().tolist()
        labels += lbl.cpu().tolist()
        accum_loss += loss.item()
    flat_epoch_preds = np.array(epoch_preds)
    flat_labels = np.array(labels)
    epoch_loss = accum_loss/len(flat_labels)
    return epoch_loss, flat_epoch_preds, flat_labels

# ==== END CODE BLOCKS ====


def main():
    # load data
    # mnli_raw = load_dataset("multi_nli")
    cola_raw = load_dataset("glue", "cola")

    # load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tok_func = get_nli_tok_func(tokenizer)
    cola_tok = cola_raw.map(tok_func, batched=True)

    # initialize datasets that contain (input, attention_mask, label)
    # train_data = BERTLikeDataset(mnli_tok['train'])
    # val_m_data = BERTLikeDataset(mnli_tok['validation_matched'])
    # val_mm_data = BERTLikeDataset(mnli_tok['validation_mismatched'])
    train_data = BERTLikeDataset(cola_tok['train'])
    val_data = BERTLikeDataset(cola_tok['validation'])
    test_data = BERTLikeDataset(cola_tok['test'])

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)

    # HYPERPARAMETERS
    lr = 1e-5
    wd = 1e-5
    n_epochs = 20
    bsz = 64
    device = 'cuda:0'
    run_name = 'bert_finetune'

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_func = nn.CrossEntropyLoss()

    # create dataloaders (batches and randomizes samples)
    train_dl = DataLoader(train_data, batch_size=bsz, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=bsz, shuffle=False)
    test_dl = DataLoader(test_data, batch_size=bsz, shuffle=False)

    # ==== RUN TRAINING ====
    model.to(device)
    unique_run_name = f"{run_name}-{current_pst_time().strftime('%Y_%m_%d-%H_%M')}"
    log_dir = f"./logs/{unique_run_name}"
    best_val_acc = 0
    with SummaryWriter(log_dir=log_dir) as log_writer:
        for epoch in range(n_epochs):
            tr_loss, tr_pred, tr_y = bert_epoch_loop(
                model, train_dl, loss_func, device=device, optimizer=optimizer)
            tr_metrics = [('loss', tr_loss),
                          ('accuracy', acc_func(tr_pred, tr_y))]
            log_metrics(log_writer, tr_metrics, epoch, data_src='train')
            print(
                f"Epoch {epoch} train: loss={tr_metrics[0][1]}, acc={tr_metrics[1][1]}")

            # run on val matched dataset
            with torch.no_grad():
                v_loss, v_pred, v_y = bert_epoch_loop(
                    model, val_dl, loss_func, device=device)
                v_acc = acc_func(v_pred, v_y)
                v_metrics = [('loss', v_loss),
                             ('accuracy', v_acc)]
                log_metrics(log_writer, v_metrics,
                            epoch, data_src='val_match')
                print(
                    f"Epoch {epoch} val_m: loss={v_metrics[0][1]}, acc={v_metrics[1][1]}")
                val_acc = (v_acc + v_acc)/2
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(),
                               f'checkpoints/{unique_run_name}.pth')
        log_writer.add_hparams({'best_val_acc': best_val_acc})


if __name__ == '__main__':
    main()
