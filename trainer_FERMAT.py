import time
import random
import math
import torch
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import re
import logging
import json
import argparse
import pandas as pd
import matplotlib.pyplot
import matplotlib.pyplot as plt
from char_level_rep import char_level_representation_encoding
logger = logging.getLogger(__name__)



# Define a custom dataset class for DROP
class DropDataset(Dataset):
    def __init__(self, examples, model, device, f_bos, f_eos):
        self.examples = examples
        self.model = model
        self.device = device
        self.f_bos = f_bos
        self.f_eos = f_eos

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        inputs = example['question']
        target = example["answer"]
        return inputs, target

    def __repr__(self):
        return f"DropDataset(examples={len(self.examples)})"

# # Define a function to calculate the accuracy of the model's predictions.
# def top_k_predictions(outputs, labels):
#   """Calculates the accuracy of the model's predictions.

#   Args:
#     outputs: The outputs of the model.
#     labels: The ground truth labels.

#   Returns:
#     The accuracy of the model's predictions.
#   """

#   predictions = torch.argmax(outputs.logits, dim=-1)
#   acc = 0
#   for i in range(len(outputs.logits)):
#     # Get the prediction tokens.
#     prediction_tokens = tokenizer.convert_ids_to_tokens(torch.argmax(outputs.logits[i], dim=-1))
#     #------------------------------------- Pratheek line of interest ---------------------#
#     # Convert logits into probability (0 to 1)
#     probability_logits = torch.nn.functional.softmax(outputs.logits[i], dim=-1)
#     top_k = 3
#     # extract top_k probabilities and token_ids
#     top_token_prob, top_token_ids = torch.topk(probability_logits, top_k)
#     # print predicted tokens with associated probability from most probable to least probable
#     for k, (token_ids, probs) in enumerate(zip(top_token_ids, top_token_prob)):
#         print(f'predictions of token {k}:')
#         for tokens, prob in zip(tokenizer.convert_ids_to_tokens(token_ids), torch.round(torch.tensor(probs), decimals=3)):
#             print(tokens, ':', prob.item())
#         print('\n')
#     #------------------------------------- Pratheek line of interest ---------------------#
              
#     label_tokens = tokenizer.convert_ids_to_tokens(labels[i])

#     # Join the prediction tokens into a string.
#     prediction_string = tokenizer.decode(torch.argmax(outputs.logits[i], dim=-1), skip_special_tokens=True)
#     label_string = tokenizer.decode(labels[i], skip_special_tokens=True)
#     if prediction_string.lower() == label_string.lower():
#       print(f'Prediction: {prediction_string}, Label: {label_string}')          
#       acc += 1

#   total = labels.size(0)
#   return acc

######## For the interactive input
def get_user_input(file_path):
    question = input("Please enter your question: ")
    answer = input("Enter the expected answer: ")
    
    # Construct a dictionary with the question, answer, and optionally the equation
    new_data = {"question": question, "answer": answer}
    
    # Convert the dictionary to a JSON string
    json_data = json.dumps(new_data)
    
    # Append the new JSON string to the file, followed by a newline character
    with open(file_path, 'a') as file:
        file.write(json_data + '\n')
    
    print(f"Your input has been added to {file_path}")
    return question, answer



def top_k_predictions(outputs, labels, filepath='top_k_predictions.txt'):
    predictions = torch.argmax(outputs.logits, dim=-1)
    acc = 0
    visualization_data = []
    answer_list = []

    with open(filepath, 'w') as file:
        token_probs = []
        for output_i in range(len(outputs.logits)):
            prediction_tokens = tokenizer.convert_ids_to_tokens(torch.argmax(outputs.logits[output_i], dim=-1))
            probability_logits = torch.nn.functional.softmax(outputs.logits[output_i], dim=-1)
            top_k = 10

            # Extract top_k probabilities and token_ids
            top_token_prob, top_token_ids = torch.topk(probability_logits, top_k, dim=-1)

            count = 0
            for token_id, prob in zip(top_token_ids, top_token_prob):
                file.write(f'Predictions for token {count}:\n')
                tokens = tokenizer.convert_ids_to_tokens(token_id)
                probs = prob.tolist()

                # Filter and round probabilities
                exclude_pattern = re.compile(r'^<extra_id_\d+>$|^<pad>$|^▁\d+$')
                filtered_token_probs = [(token, prob) for token, prob in zip(tokens, probs) if not exclude_pattern.match(token)]

                for token, prob in filtered_token_probs:
                    file.write(f'{token}: {prob:.5f}\n')

                token_probs.extend(filtered_token_probs)

                all_probabilities_sorted = sorted(token_probs, key=lambda x: x[1], reverse=True)
                file.write('\n')
                count += 1
        
        
            # if token_probs:   
            visualization_data.append(all_probabilities_sorted)
            label_tokens = tokenizer.convert_ids_to_tokens(labels[output_i])
            prediction_string = tokenizer.decode(torch.argmax(outputs.logits[output_i], dim=-1), skip_special_tokens=True)
            label_string = tokenizer.decode(labels[output_i], skip_special_tokens=True)
            if prediction_string.lower() == label_string.lower():
                acc += 1

            # return token_probs #################### For the interactive input
            answer_list.append(label_string)



    # Read the output from top_k_predictions.txt
    with open('top_k_predictions.txt', 'r') as file:
        output = file.read()

        # Split the output into questions
        questions = output.split('Predictions for token 0:')
        count = 0

        for q_idx, question in enumerate(questions):
            if not question.strip():
                continue  
                    

            # Split the question into token predictions
            token_predictions = re.split(r'Predictions for token \d+:\n', question)[1:]

            # Get the expected answer for this question
            expected_answer = answer_list[count]

            # Check if the expected answer is a decimal value
            if '.' in expected_answer:
                integer_part = expected_answer.split('.')[0]
                expected_digits = [int(d) for d in integer_part]
            else:
                expected_digits = [int(d) for d in str(expected_answer)]

            # Calculate the number of rows and columns for the subplots
            num_subplots = len(token_predictions)
            num_cols = 4  # Set the desired number of columns
            num_rows = math.ceil(num_subplots / num_cols)

            # Create a figure with multiple subplots
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(45, 20))
            axs = axs.flatten()

            # Change the font size for the y-axis tick labels
            for ax in axs:
                for label in ax.get_yticklabels():
                    label.set_fontsize(15)

            # Tokens to exclude
            exclude_tokens = {"▁", "</s>", "."}

            # Iterate over token predictions and create bar charts
            for i, prediction in enumerate(token_predictions):
                lines = prediction.strip().split('\n')
                token_probs = []
                for line in lines:
                    if ': ' in line:
                        token, prob = line.split(': ')
                        # if token not in exclude_tokens:
                        token_probs.append((token, float(prob)))

                
                token_probs_dict = dict(token_probs)

                # Add missing digits with 0.0 probability
                for digit in range(10):
                    token = str(digit)  # Convert digit to string
                    if token not in token_probs_dict:
                        token_probs_dict[token] = 0.0

                # Sort digits in increasing order
                token_probs = sorted(token_probs_dict.items(), key=lambda x: x[0])
                                


                # Extract tokens and probabilities
                tokens = [token for token, _ in token_probs]
                probs = [prob for _, prob in token_probs]

                # Convert probabilities to -log(prob)
                # log_probs = [-math.log(prob) if prob > 0 else 0 for _, prob in token_probs]
                # print(f'Probs: {probs}, Log_Probs: {log_probs}')

                # Set the y-axis limit to 1.1
                axs[i].set_ylim(0, 1.1)                

                # Create a bar chart for the current token
                positions = range(len(tokens))
                colors = ['red' if x == max(probs) else 'skyblue' for x in probs]
                bars = axs[i].bar(positions, probs, color=colors)
                axs[i].set_xticks(positions)

                x_labels = []
                for j, token in enumerate(tokens):
                    # Check if i is within the valid range of expected_digits
                    if i < len(expected_digits):
                        if token in str(expected_digits[i]):
                            x_labels.append(token + '*')
                        else:
                            x_labels.append(token)
                    else:
                        x_labels.append(token)
                
                axs[i].set_xticklabels(x_labels,fontsize=15)
                axs[i].set_xlabel('Token',fontsize=15,labelpad=20)
                axs[i].set_ylabel('-log(Probability)',fontsize=15, labelpad=20)
                axs[i].set_title(f'Token {i}',fontsize=15)

                # Add probability values on top of each bar
                for bar in bars:
                    yval = bar.get_height()
                    axs[i].text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 8), ha='center', va='bottom',fontsize=13)

            # Adjust spacing between subplots
            plt.subplots_adjust(hspace=0.4, wspace=0.3)

            # Add a main title
            
            # print(answer_list)
            plt.suptitle(f'Top Token Probabilities for Prediction Steps of Question {q_idx} - Expected Answer: {answer_list[count]}', fontsize=25)

            # Save the figure            
            plt.savefig(f'{image_dir}/top_token_probabilities_question_{q_idx}.png')
            plt.close()
            count += 1





# Define training and evaluation functions
def train(model, train_dataloader, optimizer):
    model.train()
    total_loss = 0.0
    acc = 0
    count = 0
    
    for inputs, target in train_dataloader:
        count += len(target)
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, labels=target, return_dict=True)
        # Calculate the accuracy.
        loss = outputs.loss
        loss.backward(retain_graph=True) 
        optimizer.step()
        total_loss += loss.item()
        top_k_predictions(outputs, target)

    return total_loss / len(train_dataloader), acc * 100 / count


def evaluate(model, eval_dataloader, epoch):
    model.eval()
    total_loss = 0.0
    acc = 0
    count = 0

    with torch.no_grad():
        for inputs, target in eval_dataloader:
            count += len(target)
            outputs = model(input_ids=inputs, labels=target, return_dict=True)
            loss = outputs.loss
            top_k_predictions(outputs, target)
            total_loss += loss.item()
            print(f'Evaluate ----- Accuracy: {acc}')            
    return total_loss / len(eval_dataloader), acc * 100 / count


def testing(model, test_dataloader):
    model.eval()
    total_loss = 0.0
    acc = 0
    count = 0
    with open(f'{output_dir}/pred.txt', 'w') as f:
      f.write(f'predictions /// labels /// Correct or wrong \n')

    with torch.no_grad():
        for inputs, target in test_dataloader:
            count += len(target)
            outputs = model(input_ids=inputs, labels=target, return_dict=True)
            loss = outputs.loss
            top_k_predictions(outputs, target)            
            total_loss += loss.item()                      
            for i in range(len(outputs.logits)):

                # Join the prediction tokens into a string.
                prediction_string = tokenizer.decode(torch.argmax(outputs.logits[i], dim=-1), skip_special_tokens=True)
                label_string = tokenizer.decode(target[i], skip_special_tokens=True)

                
                # Print the prediction string
                with open(f'{output_dir}/pred.txt', 'a') as f:
                
                    f.write(f"{float(prediction_string)} /// {float(label_string)} /// {'CORRECT' if float(prediction_string) == float(label_string)else 'WRONG'} \n")

                if float(prediction_string) == float(label_string):
                    acc += 1        
        print(f'Testing ----- Accuracy: {acc}/{len(target)}')  
    return total_loss / len(test_dataloader), (acc / count) * 100




def data_reader(path, model, f_bos, f_eos, device):
  DROP = []
  with open(path) as f:
    for line in f:
      problem = json.loads(line)
      problem['question'] = char_level_representation_encoding(problem['question'], tokenizer, max_length=max_length_input)['input_ids'].to(device)
      problem['answer'] = char_level_representation_encoding(problem['answer'], tokenizer, max_length=max_length_label)['input_ids'].to(device)
      DROP.append(problem)
  return DROP

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# Fine-tuning process
def fine_tune_bart(tokenizer, model_name, max_length_input, max_length_label, batch_size, num_epochs, learning_rate, output_dir, early_stopping):
    # Initialize the model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    
    # Add the special tokens to the vocabulary
    special_tokens_dict = {'additional_special_tokens': ['[F]', '[/F]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Get embeddings or input_ids of special tokens
    f_bos = tokenizer.convert_tokens_to_ids('[F]')
    f_eos = tokenizer.convert_tokens_to_ids('[/F]')

    # Replace the default embedding layer with custom embeddings
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)

    # Generate embeddings for the Dataloader
    drop_train = data_reader(f'{data_dir}/mixed_200000.train', model, f_bos, f_eos, device)
    drop_val = data_reader(f'{data_dir}/mixed_1000.dev', model, f_bos, f_eos, device)

    # Prepare the dataset
    train_dataset = DropDataset(drop_train, model, device, f_bos, f_eos)
    val_dataset = DropDataset(drop_val, model, device, f_bos, f_eos)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_acc = -1
    early_stopping_count = 0

    # # Training loop
    # for epoch in tqdm(range(num_epochs)):
    #     train_loss, train_acc = train(model, train_dataloader, optimizer)
    #     val_loss, val_acc = evaluate(model, val_dataloader, epoch)

    #     with open(f'{output_dir}/logs', 'a') as f:
    #         f.write(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc} \n")

    #     # Save the model if validation loss improves
    #     if round(val_acc,1) > round(best_val_acc,1):
    #         best_val_acc = val_acc
    #         model.save_pretrained(f'{output_dir}/checkpoint', from_pt=True)
    #         early_stopping_count = 0
    #     else:
    #         early_stopping_count +=1
    #         if early_stopping_count >= early_stopping:
    #           break

    # After training, load the model for testing
    model_test = AutoModelForSeq2SeqLM.from_pretrained(f'{output_dir}/checkpoint')
    model_test.to(device)


    for test_set in os.listdir(data_dir):
      if '.test2' in test_set:
        drop_test = data_reader(f'{data_dir}/{test_set}', model_test, f_bos, f_eos, device)
        test_dataset = DropDataset(drop_test, model, device, f_bos, f_eos)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        test_loss, test_acc = testing(model_test, test_dataloader)

        print(f"Test Loss {test_set}: {test_loss}, Test Acc {test_set}: {test_acc}")


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name',
                    type=str,
                    default='jasivan/flan-t5-base-FERMAT',
                    help='name of model as in huggingface')
parser.add_argument('--max_length_input',
                    type=int,
                    default=128,
                    help='max length of tokens in input, default 128')
parser.add_argument('--max_length_label',
                    type=int,
                    default=8,
                    help='max length of tokens in output, default 16')
parser.add_argument('--batch_size',
                    type=int,
                    default=30,
                    help='batch size, default 32')
parser.add_argument('--num_epochs',
                    type=int,
                    default=0,
                    help='number of epochs')
parser.add_argument('--lr',
                    type=float,
                    default=0.00001,
                    help='learning rate')
parser.add_argument('--data_dir',
                    type=str,
                    default = '/Users/vasupratheek/Desktop/University/Dissertation',
                    help='path to data directory')
parser.add_argument('--output_dir',
                    type=str,
                    default = 'output',
                    help='path to output directory')
parser.add_argument('--image_dir',
                    type=str,
                    default = 'images',
                    help='path to images directory')

parser.add_argument('--early_stopping',
                    type=int,
                    default = 10,
                    help='how large you want your early stopping based on val acc')
parser.add_argument('--seed',
                    type=int,
                    default = 42,
                    help='choice of random seed value')
args = parser.parse_args()

model_name = args.model_name
max_length_input = args.max_length_input
max_length_label = args.max_length_label
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.lr
data_dir = args.data_dir
output_dir = args.output_dir
image_dir = args.image_dir
early_stopping = args.early_stopping
random_seed = args.seed


use_amp = False
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
random.seed(random_seed)

# Example data for DROP dataset
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(f'{output_dir}/logs', 'w') as f:
    f.write('logs \n')

print(args)

# Load the BART tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

if __name__ == "__main__":
    # Fine-tune the BART model with the custom embeddings.
    # for test_set in os.listdir(data_dir):
        #   if '.test2' in test_set:
        #     get_user_input(f'{data_dir}/{test_set}')
    fine_tune_bart(tokenizer, model_name, max_length_input, max_length_label, batch_size, num_epochs, learning_rate, output_dir, early_stopping) 