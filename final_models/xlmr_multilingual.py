#for colab
#from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from keras_preprocessing.sequence import pad_sequences
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import os
import numpy as np
import pickle
import argparse
import time
import sys


output_file = open('output.txt', 'w')

# Redirect stdout to the file
sys.stdout = output_file

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

class XLMRClassificationTraining():
    """Finetunes XLM-R model on a classification task"""

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
        """Training loop for fine-tuning."""

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

            # loss is only output when labels are provided as input to the model
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
    
    def predict(self, predict_dataloader, tokenizer, return_probabilities=False):
        """Testing of trained checkpoint. Now also returns the original texts of the instances that were classified."""
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        probabilities = []
        original_texts = []  # List to hold the original texts of classified instances
        data_iterator = tqdm(predict_dataloader, desc="Iteration")
        softmax = torch.nn.Softmax(dim=-1)
        for step, batch in enumerate(data_iterator):
            input_ids, input_mask = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)

            logits = outputs[0]
            probs = softmax(logits)
            logits = logits.to('cpu').numpy()
            probs = probs.to('cpu').numpy()

            # Decode the input_ids for each instance in the batch and store them
            batch_texts = tokenizer.batch_decode(input_ids.to('cpu'), skip_special_tokens=True)
            original_texts.extend(batch_texts)

            for l, prob in zip(logits, probs):
                predictions.append(np.argmax(l))
                probabilities.append(prob)

        if return_probabilities:
            return predictions, probabilities, original_texts
        else:
            return predictions, original_texts

    def get_metrics(self, true, predicted):
        metrics = {'accuracy': accuracy_score(true, predicted),
                   'recall': recall_score(true, predicted, average="macro"),
                   'precision': precision_score(true, predicted, average="macro"),
                   'f1': f1_score(true, predicted, average="macro")}

        return metrics

def main():
    parser = argparse.ArgumentParser(description="Train an XLM-R model on a classification task.")
    parser.add_argument("--train_data", required=True, help="Path to the training dataset CSV file.")
    parser.add_argument("--test_data_en", required=True, help="Path to the EN test dataset CSV file.")
    parser.add_argument("--test_data_it", required=True, help="Path to the IT test dataset CSV file.")
    parser.add_argument("--test_data_slo", required=True, help="Path to the SLO test dataset CSV file.")
    parser.add_argument("--output_dir", help="Directory to save the trained model.", default="./model_output")
    parser.add_argument("--results_dir", help="Directory to save the results files.", default="./results")
    parser.add_argument("--batch_size", type=int, help="Batch size for training and evaluation.", default=32)
    parser.add_argument("--max_len", type=int, help="Maximum sequence length.", default=128)
    parser.add_argument("--epochs", type=int, help="Number of training epochs.", default=3)

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # Reading datasets
    df_train_en = pd.read_csv(args.train_data)
    df_test_en = pd.read_csv(args.test_data_en)
    df_test_it = pd.read_csv(args.test_data_it)
    df_test_slo = pd.read_csv(args.test_data_slo)
    
    
    #df_train_en = df_train_en[0:1000]
    #df_test_en =  df_test_en[0:1000]
    #df_test_it =  df_test_it[0:1000]
    #df_test_slo =  df_test_slo[0:1000]

    # Initialize tokenizer and model    
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=4)
    
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
    #Create an Instance of XLMRClassificationTraining
    training_instance = XLMRClassificationTraining(
        model=model,
        device=device,
        tokenizer=tokenizer,
        batch_size=32,
        lr=2e-5,
        train_epochs=1,
        weight_decay=0.01,
        warmup_proportion=0.1,
        adam_epsilon=1e-8 
    )

    output_dir = args.output_dir  # Directory to save the model

    # Start timing
    start_time = time.time()


    train_loss, eval_metrics = training_instance.train(train_dataloader, eval_dataloader, output_dir, save_best=True)

    # End timing
    end_time = time.time()

    # Calculate and print the training duration
    training_duration = end_time - start_time
    print(f"Training complete in {training_duration:.2f} seconds.")

    #Model evaluation EN
    test_data_en = df_test_en['Text'].values

    test_labels_en = df_test_en['Type'].values
    test_dataloader_en = prepare_data_for_testing(test_data_en, tokenizer, args.max_len, args.batch_size)

    predictions, probabilities, classified_instances = training_instance.predict(test_dataloader_en, tokenizer, return_probabilities=True)

    instances_predictions = {instance: pred for instance, pred in zip(classified_instances, predictions)}

    cm = confusion_matrix(test_labels_en, predictions)
    print(cm)
    print('Accuracy:' + str(accuracy_score(test_labels_en, predictions)))
    print('Micro F1 score:' + str(f1_score(test_labels_en, predictions, average='micro')))
    print('Macro F1 score:' + str(f1_score(test_labels_en, predictions, average='macro')))
    f1_scores = f1_score(test_labels_en, predictions, average=None)
    print("F1 '0 appropriate: '" + str(f1_scores[0]))
    print("F1 '1 inappropriate: '" + str(f1_scores[1]))
    print("F1 '2 offensive: '" + str(f1_scores[2]))
    print("F1 '3 violent: '" + str(f1_scores[3]))
    print('Recall violent class:' + str(recall_score(test_labels_en, predictions, labels=[3], average=None)[0]))

    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(test_labels_en, predictions),
        'micro_f1_score': f1_score(test_labels_en, predictions, average='micro'),
        'macro_f1_score': f1_score(test_labels_en, predictions, average='macro'),
        'f1_scores': {
            '0_appropriate': f1_scores[0],
            '1_inappropriate': f1_scores[1],
            '2_offensive': f1_scores[2],
            '3_violent': f1_scores[3]
        },
        'recall_3_violent': recall_score(test_labels_en, predictions, labels=[3], average=None)[0]
    }

    # Save metrics to a pickle file in the specified directory
    metrics_file_path = os.path.join(args.results_dir, 'metrics_xlmr_en.pkl')
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(metrics, file)

    # Save predictions to a pickle file in the specified directory
    predictions_file_path = os.path.join(args.results_dir, 'predictions_xlmr_en.pkl')
    with open(predictions_file_path, 'wb') as file:
        pickle.dump(instances_predictions, file)

    
    #Model evaluation IT

    test_data_it = df_test_it['Testo'].values

    test_labels_it = df_test_it['Tipo'].values
    test_dataloader_it = prepare_data_for_testing(test_data_it, tokenizer, args.max_len, args.batch_size)

    predictions, probabilities, classified_instances = training_instance.predict(test_dataloader_it, tokenizer, return_probabilities=True)

    instances_predictions = {instance: pred for instance, pred in zip(classified_instances, predictions)}

    cm = confusion_matrix(test_labels_it, predictions)
    print(cm)
    print('Accuracy:' + str(accuracy_score(test_labels_it, predictions)))
    print('Micro F1 score:' + str(f1_score(test_labels_it, predictions, average='micro')))
    print('Macro F1 score:' + str(f1_score(test_labels_it, predictions, average='macro')))
    f1_scores = f1_score(test_labels_it, predictions, average=None)
    print("F1 '0 appropriate: '" + str(f1_scores[0]))
    print("F1 '1 inappropriate: '" + str(f1_scores[1]))
    print("F1 '2 offensive: '" + str(f1_scores[2]))
    print("F1 '3 violent: '" + str(f1_scores[3]))
    print('Recall violent class:' + str(recall_score(test_labels_it, predictions, labels=[3], average=None)[0]))

    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(test_labels_it, predictions),
        'micro_f1_score': f1_score(test_labels_it, predictions, average='micro'),
        'macro_f1_score': f1_score(test_labels_it, predictions, average='macro'),
        'f1_scores': {
            '0_appropriate': f1_scores[0],
            '1_inappropriate': f1_scores[1],
            '2_offensive': f1_scores[2],
            '3_violent': f1_scores[3]
        },
        'recall_3_violent': recall_score(test_labels_it, predictions, labels=[3], average=None)[0]
    }

   # Save metrics to a pickle file in the specified directory
    metrics_file_path = os.path.join(args.results_dir, 'metrics_xlmr_it.pkl')
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(metrics, file)

    # Save predictions to a pickle file in the specified directory
    predictions_file_path = os.path.join(args.results_dir, 'predictions_xlmr_it.pkl')
    with open(predictions_file_path, 'wb') as file:
        pickle.dump(instances_predictions, file)

    
    #Model evaluation SLO

    test_data_slo = df_test_slo['besedilo'].values

    test_labels_slo = df_test_slo['vrsta'].values
    test_dataloader_slo = prepare_data_for_testing(test_data_slo, tokenizer, args.max_len, args.batch_size)

    predictions, probabilities, classified_instances = training_instance.predict(test_dataloader_slo, tokenizer, return_probabilities=True)

    instances_predictions = {instance: pred for instance, pred in zip(classified_instances, predictions)}

    cm = confusion_matrix(test_labels_slo, predictions)
    print(cm)
    print('Accuracy:' + str(accuracy_score(test_labels_slo, predictions)))
    print('Micro F1 score:' + str(f1_score(test_labels_slo, predictions, average='micro')))
    print('Macro F1 score:' + str(f1_score(test_labels_slo, predictions, average='macro')))
    f1_scores = f1_score(test_labels_slo, predictions, average=None)
    print("F1 '0 appropriate: '" + str(f1_scores[0]))
    print("F1 '1 inappropriate: '" + str(f1_scores[1]))
    print("F1 '2 offensive: '" + str(f1_scores[2]))
    print("F1 '3 violent: '" + str(f1_scores[3]))
    print('Recall violent class:' + str(recall_score(test_labels_slo, predictions, labels=[3], average=None)[0]))

    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(test_labels_slo, predictions),
        'micro_f1_score': f1_score(test_labels_slo, predictions, average='micro'),
        'macro_f1_score': f1_score(test_labels_slo, predictions, average='macro'),
        'f1_scores': {
            '0_appropriate': f1_scores[0],
            '1_inappropriate': f1_scores[1],
            '2_offensive': f1_scores[2],
            '3_violent': f1_scores[3]
        },
        'recall_3_violent': recall_score(test_labels_slo, predictions, labels=[3], average=None)[0]
    }

    # Save metrics to a pickle file in the specified directory
    metrics_file_path = os.path.join(args.results_dir, 'metrics_xlmr_slo.pkl')
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(metrics, file)

    # Save predictions to a pickle file in the specified directory
    predictions_file_path = os.path.join(args.results_dir, 'predictions_xlmr_slo.pkl')
    with open(predictions_file_path, 'wb') as file:
        pickle.dump(instances_predictions, file)

    output_file.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()

# Example:
# python your_script_name.py --train_data path/to/train.csv --test_data_en path/to/test_en.csv --test_data_it path/to/test_it.csv --test_data_slo path/to/test_slo.csv --output_dir path/to/output --results_dir path/to/results --batch_size 32 --max_len 128 --epochs 3
