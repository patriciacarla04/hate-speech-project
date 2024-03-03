#for colab
#from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import os
import numpy as np
from krippendorf import IntervalKAlpha
import pickle
import argparse


def prepare_data_for_training(data, labels, tokenizer, max_len, batch_size):
    """REFACTOR: MAKE TWO FUNCTIONS - ONE PREPARES DATA, THE OTHER CREATES A DATALOADER"""
    #sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in data]
    truncated_sentences = [sentence[:(max_len - 2)] for sentence in tokenized_sentences]
    truncated_sentences = [["[CLS]"] + sentence + ["[SEP]"] for sentence in truncated_sentences]
    print("Example of tokenized sentence:")
    print(truncated_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in truncated_sentences]
    print("Printing encoded sentences:")
    print(input_ids[0])
    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def prepare_data_for_testing(data, tokenizer, max_len, batch_size):
    """REFACTOR: MAKE TWO FUNCTIONS - ONE PREPARES DATA, THE OTHER CREATES A DATALOADER"""
    #sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in data]
    truncated_sentences = [sentence[:(max_len - 2)] for sentence in tokenized_sentences]
    truncated_sentences = [["[CLS]"] + sentence + ["[SEP]"] for sentence in truncated_sentences]
    print("Example of tokenized sentence:")
    print(truncated_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in truncated_sentences]
    print("Printing encoded sentences:")
    print(input_ids[0])
    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks)
    sampler = SequentialSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader

class BertClassificationTraining():
    """Finetunes multilingual BERT model on a classification task"""

    def __init__(self, model, device, tokenizer, batch_size=32, lr=2e-5, train_epochs=3, weight_decay=0.01,
                 warmup_proportion=0.1, adam_epsilon=1e-8):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.learning_rate = lr
        self.train_epochs = train_epochs
        self.weight_decay = weight_decay
        self.warmup_proportion = warmup_proportion
        self.adam_epsilon = adam_epsilon

    def train(self, train_dataloader, eval_dataloader, output_dir, save_best=False, eval_metric='f1'):
        """Training loop for bert fine-tuning."""

        t_total = len(train_dataloader) * self.train_epochs
        warmup_steps = len(train_dataloader) * self.warmup_proportion
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        train_iterator = trange(int(self.train_epochs), desc="Epoch")
        #model = self.model
        self.model.to(self.device)
        tr_loss_track = []
        eval_metric_track = []
        output_filename = os.path.join(output_dir, 'pytorch_model.bin')
        metric = float('-inf')

        for _ in train_iterator:
            self.model.train()
            self.model.zero_grad()
            tr_loss = 0
            nr_batches = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                tr_loss = 0
                input_ids, input_mask, labels = batch
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=input_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                scheduler.step()
                tr_loss += loss.item()
                nr_batches += 1
                self.model.zero_grad()

            print("Evaluating the model on the evaluation split...")
            metrics = self.evaluate(eval_dataloader)
            eval_metric_track.append(metrics)
            if save_best:
                if metric < metrics[eval_metric]:
                    self.model.save_pretrained(output_dir)
                    torch.save(self.model.state_dict(), output_filename)
                    print("The new value of " + eval_metric + " score of " + str(metrics[eval_metric]) + " is higher then the old value of " +
                          str(metric) + ".")
                    print("Saving the new model...")
                    metric = metrics[eval_metric]
                else:
                    print(
                        "The new value of " + eval_metric + " score of " + str(metrics[eval_metric]) + " is not higher then the old value of " +
                        str(metric) + ".")

            tr_loss = tr_loss / nr_batches
            tr_loss_track.append(tr_loss)

        if not save_best:
            self.model.save_pretrained(output_dir)
            # tokenizer.save_pretrained(output_dir)
            torch.save(self.model.state_dict(), output_filename)

        return tr_loss_track, eval_metric_track

    def evaluate(self, eval_dataloader):
        """Evaluation of trained checkpoint."""
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        true_labels = []
        data_iterator = tqdm(eval_dataloader, desc="Iteration")
        for step, batch in enumerate(data_iterator):
            input_ids, input_mask, labels = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)

            # loss is only output when labels are provided as input to the model ... real smooth
            logits = outputs[0]
            print(type(logits))
            logits = logits.to('cpu').numpy()
            label_ids = labels.to('cpu').numpy()

            for label, logit in zip(label_ids, logits):
                true_labels.append(label)
                predictions.append(np.argmax(logit))

        # print(predictions)
        # print(true_labels)
        metrics = self.get_metrics(true_labels, predictions)
        return metrics

    def predict(self, predict_dataloader, return_probabilities=False):
        """Testing of trained checkpoint.
        CHANGE SO IT DOES NOT TAKE DATALOADER WITH LABELS BECAUSE WE DON'T NEED THEM"""
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        probabilities = []
        # true_labels = []
        data_iterator = tqdm(predict_dataloader, desc="Iteration")
        softmax = torch.nn.Softmax(dim=-1)
        for step, batch in enumerate(data_iterator):
            input_ids, input_mask = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)

            # loss is only output when labels are provided as input to the model
            logits = outputs[0]
            #print(type(logits))
            probs = softmax(logits)
            logits = logits.to('cpu').numpy()
            probs = probs.to('cpu').numpy()
            # label_ids = labels.to('cpu').numpy()

            for l, prob in zip(logits, probs):
                # true_labels.append(label)
                predictions.append(np.argmax(l))
                probabilities.append(prob)

        # print(predictions)
        # print(true_labels)
        # metrics = get_metrics(true_labels, predictions)
        if return_probabilities == False:
            return predictions
        else:
            return predictions, probabilities

    def get_metrics(self, true, predicted):
        metrics = {'accuracy': accuracy_score(true, predicted),
                   'recall': recall_score(true, predicted, average="macro"),
                   'precision': precision_score(true, predicted, average="macro"),
                   'f1': f1_score(true, predicted, average="macro")}

        return metrics

def main():
    parser = argparse.ArgumentParser(description="Train a BERT model on a classification task.")
    parser.add_argument("--train_data", required=True, help="Path to the training dataset CSV file.")
    parser.add_argument("--test_data", required=True, help="Path to the test dataset CSV file.")
    parser.add_argument("--output_dir", help="Directory to save the trained model.", default="./model_output")
    parser.add_argument("--batch_size", type=int, help="Batch size for training and evaluation.", default=32)
    parser.add_argument("--max_len", type=int, help="Maximum sequence length.", default=128)
    parser.add_argument("--epochs", type=int, help="Number of training epochs.", default=3)

    args = parser.parse_args()

    # Reading datasets
    df_train_en = pd.read_csv(args.train_data)
    df_test_en = pd.read_csv(args.test_data)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    
    data = df_train_en['Text'].values
    labels = df_train_en['Type'].values

    # Split the data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(data, labels, test_size=.1)

    # Prepare the DataLoader for training and testing
    train_dataloader = prepare_data_for_training(train_texts, train_labels, tokenizer, args.max_len, args.batch_size)
    eval_dataloader = prepare_data_for_training(val_texts, val_labels, tokenizer, args.max_len, args.batch_size)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training and evaluation
    #Create an Instance of BertClassificationTraining
    training_instance = BertClassificationTraining(
        model=model,
        device=device,
        tokenizer=tokenizer,
        batch_size=32,
        lr=2e-5,
        train_epochs=3,
        weight_decay=0.01,
        warmup_proportion=0.1,
        adam_epsilon=1e-8 
    )

    output_dir = args.output_dir  # Directory to save the model

    train_loss, eval_metrics = training_instance.train(train_dataloader, eval_dataloader, output_dir, save_best=True)

    print("Training complete.")

    #Model evaluation
    alpha_metric = IntervalKAlpha()

    test_data = df_test_en['Text'].values

    test_labels = df_test_en['Type'].values
    test_dataloader = prepare_data_for_testing(test_data, tokenizer, args.max_len, args.batch_size)

    predictions, probabilities = training_instance.predict(test_dataloader, return_probabilities=True)

    cm = confusion_matrix(test_labels, predictions)
    alpha = alpha_metric.alpha_score(cm)
    print(cm)
    print('Accuracy:' + str(accuracy_score(test_labels, predictions)))
    print("Interval Alpha: {}".format(alpha))
    print('Micro F1 score:' + str(f1_score(test_labels, predictions, average='micro')))
    print('Macro F1 score:' + str(f1_score(test_labels, predictions, average='macro')))
    f1_scores = f1_score(test_labels, predictions, average=None)
    print("F1 '0 appropriate: '" + str(f1_scores[0]))
    print("F1 '1 inappropriate: '" + str(f1_scores[1]))
    print("F1 '2 offensive: '" + str(f1_scores[2]))
    print("F1 '3 violent: '" + str(f1_scores[3]))
    print('Recall violent class:' + str(recall_score(test_labels, predictions, labels=[3], average=None)[0]))

    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(test_labels, predictions),
        'interval_alpha': alpha,
        'micro_f1_score': f1_score(test_labels, predictions, average='micro'),
        'macro_f1_score': f1_score(test_labels, predictions, average='macro'),
        'f1_scores': {
            '0_appropriate': f1_scores[0],
            '1_inappropriate': f1_scores[1],
            '2_offensive': f1_scores[2],
            '3_violent': f1_scores[3]
        },
        'recall_3_violent': recall_score(test_labels, predictions, labels=[3], average=None)[0]
    }

    #Save metrics to a pickle file
    metrics_file_path = 'metrics_bert_en_duplicate_all.pkl'
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(metrics, file)

    # Save predictions to pickle file
    predictions_file_path = 'predictions_bert_en_duplicate_all.pkl'
    with open(predictions_file_path, 'wb') as file:
        pickle.dump(predictions, file)

if __name__ == "__main__":
    main()
