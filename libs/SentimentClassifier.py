from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer
import torch
import numpy as np
import datetime
import time
from torch.optim import AdamW
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score

label_mapping = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_model(
        model: BertForSequenceClassification,
        train_dataloader: DataLoader,
        optimizer: AdamW,
        scheduler: get_linear_schedule_with_warmup,
        device="cpu"):
    t0 = time.time()
    model.train()
    model.to(device)
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader):
        elapsed = format_time(time.time() - t0)
#        if step % 40 == 0 and not step == 0:
        print(f"Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}")
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        train_output = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = train_output.loss
        logits = train_output.logits

        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print("")
    print(f"  Average training loss: {avg_train_loss:.2f}")
    print(f"  Training epoch took: {training_time}")


def eval_model(
        model: BertForSequenceClassification,
        eval_dataloader: DataLoader,
        training_stats: list,
        device="cpu"):
    t0 = time.time()
    model.eval()
    model.to(device)
    total_eval_accuracy = 0
    total_eval_loss = 0
    test_f1 = 0.0
    confusion_matrix = torch.zeros((3, 3))
    for batch in eval_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            eval_output = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
            loss = eval_output.loss
            logits = eval_output.logits

            _, preds = torch.max(logits, 1)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            test_f1 += f1_score(label_ids, preds, average='macro')
            for t, p in zip(b_labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        total_eval_loss += loss.item()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    avg_val_accuracy = total_eval_accuracy / len(eval_dataloader)
    avg_val_f1 = test_f1 / len(eval_dataloader)
    avg_val_loss = total_eval_loss / len(eval_dataloader)

    validation_time = format_time(time.time() - t0)

    print(f"  Accuracy: {avg_val_accuracy:.2f}")
    print(f"  F1 Score: {avg_val_f1:.2f}")
    print(confusion_matrix)
    print(f"  Validation Loss: {avg_val_loss:.2f}")
    print(f"  Validation took: {validation_time}")
    training_stats.append(
        {
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Validation Time': validation_time,
            'Valid. F1': avg_val_f1,
            'Valid. Confusion Matrix': confusion_matrix
        }
    )

def k_cross_fold_validation(dataset: TensorDataset, k=5, epochs=2, batch_size=16, device="cpu"):
    kfold = KFold(n_splits=k, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        training_stats = []
        print(f"Fold: {fold + 1}/{k}")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False,
        )

        optimizer = AdamW(model.parameters(),
                          lr=1e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        train_dataset = Subset(dataset, train_ids)
        val_dataset = Subset(dataset, val_ids)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=RandomSampler(train_dataset)
                                      )
        validation_dataloader = DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           sampler=SequentialSampler(val_dataset)
                                           )
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=len(train_dataloader) * epochs)
        optimizer.zero_grad()
        print(f"{len(train_dataset)} training samples'")
        print(f"{len(val_dataset)} validation samples")
        for epoch in range(epochs):
            print("")
            print(f"======== Epoch {epoch + 1} / {epochs} ========")
            print("Training...")
            train_model(model, train_dataloader, optimizer, scheduler, device)
            print("")
            print("Running Validation...")
            eval_model(model, validation_dataloader,training_stats, device)

def evaluation(train_dataloader, validation_dataloader, epochs=2, device="cpu"):
    training_stats = []
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
    )
    print(f"{len(train_dataloader)} training samples'")
    print(f"{len(validation_dataloader)} validation samples")
    optimizer = AdamW(model.parameters(),
                      lr=1e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * epochs)
    optimizer.zero_grad()
    for epoch in range(epochs):
        print("")
        print(f"======== Epoch {epoch + 1} / {epochs} ========")
        print("Training...")
        train_model(model, train_dataloader, optimizer, scheduler, device)
        print("")
        print("Running Validation...")
        eval_model(model, validation_dataloader, training_stats, device)

def tweet_pipeline(tokenizer,tweet, input_ids, attention_masks):
    encoded_dict = tokenizer.encode_plus(tweet,
                                         add_special_tokens = True,
                                         padding= 'max_length',
                                         return_attention_mask = True,
                                         return_tensors = 'pt')
    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])

def preprocess(data,batch_size):
    input_ids = []
    attention_masks = []
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    labels = data["Label"].apply(lambda x: label_mapping[x]).tolist()
    data["Tweet"].apply(lambda tweet: tweet_pipeline(tokenizer,tweet,input_ids,attention_masks))
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)