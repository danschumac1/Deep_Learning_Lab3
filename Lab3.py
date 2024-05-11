#region # INTRO AND SET UP
# =============================================================================
# INTRODUCTION
# =============================================================================

# In this lab, we will learn how AI can be developed to detect cyberbullying. 
# We will use a publicly available dataset of cyberbullying texts, 
# and train an AI model on this dataset to automatically detect cyberbullying text. 

# You will learn:
# 1. AI development process
# 2. Train and test your own AI for cyberbullying detection
# 3. Run AI on your own samples
# 4. Hypterpaprameter tuning to improve mode


# =============================================================================
# # PRELIMINARIES
# =============================================================================
# from IPython.display import HTML
# shell = get_ipython()

# def adjust_font_size():
#   display(HTML('''<style>
#     body {
#       font-size: 20px;
#     }
#   '''))

# if adjust_font_size not in shell.events.callbacks['pre_execute']:
#   shell.events.register('pre_execute', adjust_font_size)

  # hide warnings

#endregion
#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers import AdamW

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

# change the device to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
#endregion
#region # DATA PREPROCESSING
# =============================================================================
# DATA PREPROCESSING
# =============================================================================
# Dowload cyberbullying speech dataset
#dowload the main dataset 
main_df = pd.read_csv('./DeepLearning/Lab-3/CyberbullyingLab1/formspring_dataset.csv', sep = '\t')

# how many smaples
print('Total number of samples:', main_df.shape)
main_df = main_df.sample(n = main_df.shape[0])
main_df = main_df[['text', 'label']]

# take a peek
main_df.head()

# While training AI, datasets are divided into three parts: 
# training dataset,  testing dataset and validation dataset.
# create splits
#  divide the dataset into non-cyberbullying and cyberbullying samples
o_class = main_df.loc[main_df.label == 0, :]
l_class = main_df.loc[main_df.label == 1, :]

# create train, val and test splits
train_val = main_df.iloc[:int(main_df.shape[0] * .80)]
test_df = main_df.iloc[int(main_df.shape[0] * .80):]
train_df = train_val.iloc[:int(train_val.shape[0] * .80)]
val_df = train_val.iloc[int(train_val.shape[0] * .80):]

#print(train.shape, val.shape, test.shape)
print('\nTraining set:\n', train_df.label.value_counts())
print('\nValidation set:\n', val_df.label.value_counts())
print('\nTest set:\n', test_df.label.value_counts())
#endregion
#region # TOKENIZATION
# =============================================================================
# Tokenization
# =============================================================================
# Let's use a tokenizer. This is the first step in NLP
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#  Let's see how the tokenizer works
sentences = "the cat sat on the mat"

tokens = tokenizer.tokenize(sentences)
for token in tokens:
    print(token)
#endregion
#region # TASK 1
# =============================================================================
# TASK 1
# Add code below to preprocess the following cyberbullying text, 
# and include the generated tokens in your report.
# =============================================================================
# TODO: Enter your code here
#  Let's see how the tokenizer works
sentences = "Harlem shake is just an excuse to go full retard for 30 seconds"

tokens = tokenizer.tokenize(sentences)
for token in tokens:
    print(token)

# With tokenizer, now we can prepare the input data that the AI model needs
#endregion
#region # PREPARE DATASET
# Prepare the dataset
class CyberbullyingDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = df.text.to_list()
        self.label = df.label.to_list()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

        inputs = self.tokenizer.encode_plus(
            text, # Sentence to encode.
            None, # Add another sequence to the inputs. 'None' means no other sequence is added.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = self.max_len, # Pad & truncate all sentences.
            pad_to_max_length = True, # Pad all samples to the same length.
            truncation = True, # Truncate all samples to the same length.
            return_token_type_ids = False,
            return_tensors = 'pt' # Return pytorch tensors.
        )
        label = torch.tensor(self.label[index], dtype = torch.long)

        return {'text': text,
                'input_ids': inputs['input_ids'].flatten(),
                'attention_mask': inputs['attention_mask'].flatten(),
                'label': label}

# The number of unique words in the vocabulary and the number of labels
VOCAB_SIZE = tokenizer.vocab_size
NUM_LABELS = train_df.label.nunique()
print("The number of unique words in the vocabulary:", VOCAB_SIZE)
print("The number of labels:", NUM_LABELS)

# In order to make the model understand both cyberbullying and non-cyberbullying data, we typically balance the datasets.

# Build a balanced dataset
def balence_data(dataframe):
    o_class = dataframe.loc[dataframe.label == 0, :]
    l_class = dataframe.loc[dataframe.label == 1, :]
    o_class = o_class.sample(n = l_class.shape[0])
    dataframe = pd.concat([o_class, l_class], axis = 0)
    dataframe = dataframe.sample(n = dataframe.shape[0])
    return dataframe

train_df = balence_data(train_df)
val_df = balence_data(val_df)
test_df = balence_data(test_df)

print('\nTraining set:\n', train_df.label.value_counts())
print('\nValidation set:\n', val_df.label.value_counts())
print('\nTest set:\n', test_df.label.value_counts())

# We need iterators to step through our dataset.

# Normally, we prepare the dataset with batches, it can help us to train the model faster.
MAX_LEN = 120
BATCH_SIZE = 32

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CyberbullyingDataset(df, tokenizer, max_len)
    return DataLoader(ds, batch_size = batch_size)

# Create the dataloaders
train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

print("After we build the dataloaders, we can see the number of batches in each dataloader. It means we can train the model with {} samples in each time.".format(BATCH_SIZE))
print("The number of batches in the training dataloader:", len(train_data_loader))
print("The number of batches in the validation dataloader:", len(val_data_loader))
print("The number of batches in the test dataloader:", len(test_data_loader))

#endregion
#region # ITERATORS

# =============================================================================
# We need iterators to step through our dataset.
# =============================================================================
# Let's define some hyperparameters for our AI model, you can change them to adjust the performance of the model.
# Lets define some hyperparameters
N_EPOCHS = 5 # The number of epochs
LEARNING_RATE = 2e-5 # The learning rate
Num_classes = 2 # The number of classes

# Let's instantiate our AI model.

# Download the tokenizer and model
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
bert_model = bert_model.to(device)


# Define the model
class CyberbullyingDetecter(nn.Module):
    def __init__(self, bert_model, Num_classes):
            super(CyberbullyingDetecter, self).__init__()
            self.bert = bert_model
            self.drop = nn.Dropout(p=0.3)
            self.out = nn.Linear(self.bert.config.hidden_size, Num_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )['pooler_output']
        output = self.drop(pooled_output)
        return self.out(output)

# Create the model
model = CyberbullyingDetecter(bert_model, Num_classes)
model = model.to(device)




# Now, let's define and the loss function.

# Define and the loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)

# Define some functions for model training
# Let's define the training and testing procedures for our AI model
# Lets define our training steps
def accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1).flatten()
    labels = labels.flatten()
    return torch.sum(preds == labels) / len(labels)

def train(model, data_loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for d in tqdm(data_loader):
        inputs_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['label'].to(device)
        # print(inputs_ids.shape, attention_mask.shape, label.shape)
        outputs = model(inputs_ids, attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

# Lets define our testing steps
def evaluate(model, data_loader, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for d in data_loader:
            inputs_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)
            outputs = model(inputs_ids, attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

# define a function for evaluation
def predict_cb(sentence):
    sentence = str(sentence)
    sentence = ' '.join(sentence.split())
    inputs = tokenizer.encode_plus(
        sentence,  # Sentence to encode.
        None,  # Add another sequence to the inputs. 'None' means no other sequence is added.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=MAX_LEN,  # Pad & truncate all sentences.
        pad_to_max_length=True,  # Pad all samples to the same length.
        truncation=True,  # Truncate all samples to the same length.
        return_token_type_ids=True  # Return token_type_ids
    )
    output = model(torch.tensor(inputs['input_ids']).unsqueeze(0).to(device), torch.tensor(inputs['attention_mask']).unsqueeze(0).to(device))
    # print(output)
    preds, ind = torch.max(F.softmax(output, dim=-1), 1)
    if ind.item() == 1:
        return preds, ind, 'Cyberbullying detected.'
    else:
        return preds, ind, 'Cyberbullying not detected.'


# =============================================================================
# ### Training Process
# =============================================================================


import os
import torch

model_path = './models/model_1.pth'

# Check if the model has already been saved
if os.path.exists(model_path):
    # Load the model state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print("Loaded saved model.")
else:
    # If the model hasn't been saved, train it
    print("Training model...")
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_data_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_data_loader, criterion)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
    
    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved.")
#endregion
#region # TASK 2
# =============================================================================
# TASK 2
# After training, what is the training accuracy that your model achieves? 
# =============================================================================
#TODO: add your code below to print the final training accuracy out

# test your CNN on the test set
# Note: If your accuracy is low, you need to further train your CNN.
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

y_test = []
y_test_predictions = []

net.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU if available
    y_test.extend([i.item() for i in labels])

    outputs = net(inputs)
    y_test_predictions.extend(torch.argmax(i).item() for i in outputs)

# print accuracy, prec, rec, f1-score here
print(
    f'accuracy_score: {accuracy_score(y_test, y_test_predictions)}',
    f"\nprecision_score (macro): {precision_score(y_test, y_test_predictions, average='macro', zero_division=0)}",
    f"\nrecall_score (macro): {recall_score(y_test, y_test_predictions, average='macro', zero_division=0)}",
    f"\nf1_score (macro): {f1_score(y_test, y_test_predictions, average='macro', zero_division=0)}"
)
#endregion
#region # TASK 3
# =============================================================================
# TASK 3
# Let's review the previous code then finish the next code cell
# =============================================================================

#TODO: complete the code below
# ..., ... = evaluate(...)
# evaluate(model, data_loader, criterion):

# print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')


### Deployment

# Example 1: "Hello World!"
text = 'hello world!'
ret = predict_cb(text)
print("Sample prediction: ", ret[2], f'Confidence: {ret[0].item() * 100:.2f}%')
#endregion
#region # TASK 4
# =============================================================================
# TASK 4
# Use the samples in this file and your model to detect the cyberbullying samples
# =============================================================================
samples = {1:" you guys are a bunch of losers, fuck you",
 2: "I'm never going to see your little pathetic self again",
 3: "She looks really nice today!"}

#TODO: complete the code below
text1 = ...
ret1 = ...
print(...)

print("=========================================")

text2 = ...
ret2 = ...
print(...)

print("=========================================")

text3 = ...
ret3 = ...
print(...)

#@title Input your own sentence to check the prediction.

My_Sentence = 'we are in the lab1' #@param {type:"string"}

ret = predict_cb(My_Sentence)
print("===========The Model Prediciton is===========")
print("The input sentence is: ", ret[2], f'Confidence: {ret[0].item() * 100:.2f}%')
#endregion
#region # HYPER PARAMETER TUNING
# =============================================================================
# HYPER PARAMETER TUNING
# =============================================================================
# @title A fast training function
def train_model(model, train_data_loader, val_data_loader, number_of_epochs, learning_rate, verbose=True):
    """
    Trains our AI model and plots the learning curve
    Arguments:
        model: model to be trained
        train_iterator: an iterator over the training set
        validation_iterator: an iterator over the validation set
        number_of_epochs: The number of times to go through our entire dataset
        optimizer: the optimization function, defaults to None
        criterion: the loss function, defaults to None
        learning_rate: the learning rate, defaults to 0.001
        verbose: Boolean - whether to print accuracy and loss
    Returns:
        learning_curve: Dictionary - contains variables for plotting the learning curve
    """

    # initialize variables for plotting
    epochs = [i for i in range(number_of_epochs)]
    train_losses = []
    validation_losses = []
    validation_accs = []

    # define the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    criterion = nn.CrossEntropyLoss().to(device)

    model = model.to(device)

    # train the model
    for epoch in range(number_of_epochs):
        train_loss, train_acc = train(model, train_data_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_data_loader, criterion)
        train_losses.append(train_loss)
        validation_losses.append(valid_loss)
        validation_accs.append(valid_acc)
        if verbose:
            print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    test_loss, test_acc = evaluate(model, test_data_loader, criterion)
    if verbose:
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
    print()

    epochs = np.asarray(epochs)
    train_losses = np.asarray(train_losses)
    validation_losses = np.asarray(validation_losses)
    validation_accs = np.asarray(validation_accs)

    learning_curve = {
        'epochs': epochs,
        'train_losses': train_losses,
        'validation_losses': validation_losses,
        'validation_accs': validation_accs,
        'learning_rate': learning_rate,
    }

    return learning_curve

# =============================================================================
# TASK 5 
# Compare different training epochs with 2, 10. You can try more different setttings and find a suitable epoch number
# =============================================================================
training_epochs = [2, 10]
learning_curve = {}

for i, epoch in enumerate(training_epochs,1):
    print(f'Training for {epoch} epochs')
    # Initialize the model
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(device)
    model = CyberbullyingDetecter(bert_model, Num_classes).to(device)
    # Train the model
    learning_curve[i] = train_model(model, train_data_loader, val_data_loader, epoch, 2e-5, verbose=True)
    print('training complete!')
for i, epoch in enumerate(training_epochs,1):
    plt.plot(learning_curve[i]['epochs'], learning_curve[i]['train_losses'], label=f'Training Loss (epochs = {epoch})')
    plt.plot(learning_curve[i]['epochs'], learning_curve[i]['validation_losses'], label=f'Validation Loss (epochs = {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # =============================================================================
    # TASK 6
    # Compare different learning rates with 0.1, 1e-3 and 1e-5. You can try with your own settings and find the best learning rate
    # =============================================================================
    #Learning rate needs to be chosen carefully in order for gradient descent to work properply. 
    # How quickly we update the parameters of our models is determined by the learning rate. 
    #If we choose the learning rate to be too small, 
    # we may need a lot more iteration to converge to the optimal values. 
    #If we choose the learning rate to be too big, we may go past our optimal values. 
    #So, it is important to choose the learning rate carefully.
learning_rates = [0.01, 1e-3, 1e-5]
learning_curve = {}

for i, lr in enumerate(learning_rates,1):
    print('Learning rate:', lr)
    # Initialize the model
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(device)
    model = CyberbullyingDetecter(bert_model, Num_classes).to(device)
    # Train the model
    learning_curve[i] = train_model(model, train_data_loader, val_data_loader, 5, lr, verbose=True)
    print('Training complete!')

for i, lr in enumerate(learning_rates,1):
    plt.plot(learning_curve[i]["epochs"], np.squeeze(learning_curve[i]["train_losses"]), label=learning_curve[i]["learning_rate"])

plt.ylabel('loss')
plt.xlabel('epochs')

legend = plt.legend(loc='best', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()



#endregion
#region # TASK 7
# =============================================================================
# Task 7
# We experimented with different hyperparameters in this lab, what can you conclude about training AI models? Specifically, 
# what are your observations about the model before Vs. after hyperparameter tuning?
# =============================================================================
#endregion