import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model import Starhub
import torch.optim as optim

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# columns => ['repository', 'year', 'code_commits_diff', 'text_commits_diff', 'main_language', 'contributors', 'commits',
#             'issues', 'pull_requests', 'releases', 'current_stars', 'estimated_stars']

df = pd.read_csv('data/repository_data.csv')

df = df[df['estimated_stars'] != 0]
df = df.sort_values(by=['repository', 'year'])
def update_current_stars(group):
    for i in range(1, len(group)):
        group.iloc[i, group.columns.get_loc('current_stars')] = group.iloc[i - 1]['current_stars'] + group.iloc[i - 1]['estimated_stars']
    return group
df = df.groupby('repository').apply(update_current_stars)
df['current_stars'] = df['current_stars'].astype(int)
df['estimated_stars'] = df['current_stars'] + df['estimated_stars']
df.rename(columns = {'estimated_stars': 'future_stars'}, inplace=True)


df = df.iloc[:10] # FOR TEST


df['main_language'] = df['main_language'].fillna('Unknown')
# language_counts = df['main_language'].value_counts()
# print(language_counts)
label_encoder = LabelEncoder()
df['main_language'] = label_encoder.fit_transform(df['main_language'])

df = df.drop(['contributors'], axis=1)

# Label Encoding for 'year'
label_enc = LabelEncoder()
df['year_encoded'] = label_enc.fit_transform(df['year'])

# Standardize numerical features
numerical_features = ['commits', 'issues', 'pull_requests', 'releases']
# scaler = StandardScaler()
# df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Standardize each numerical feature separately
for feature in numerical_features:
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])

# df = df[['repository', 'year', 'year_encoded', 'main_language', 'commits',
#          'issues', 'pull_requests', 'releases', 'current_stars', 'future_stars']]
# print(df.head(10))
# pouya()

# print(df['text_commits_diff'].values[0])
# print(df['code_commits_diff'].values[0])
# pouya()

# Split your features according to 'metadata', 'text', and 'code'
metadata_columns = ['year_encoded', 'main_language', 'commits', 'issues', 'pull_requests', 'releases']
text_columns = ['text_commits_diff']
code_columns = ['code_commits_diff']
current_stars_column = 'current_stars'
target_column = 'future_stars'

# Splitting data into features and target
X_metadata = df[metadata_columns]
X_text = df[text_columns]
X_code = df[code_columns]
current_stars = df[current_stars_column]
y = df[target_column]

from transformers import BertTokenizer, RobertaTokenizer

# Load the tokenizer for 'bert-base-uncased'
tokenizer_text = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the tokenizer for 'microsoft/codebert-base' which is based on RoBERTa
tokenizer_code = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

def tokenize_bert(data, tokenizer, max_length=512):
    return tokenizer.batch_encode_plus(
        data.tolist(),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# Convert columns to strings and handle NaN values
df['text_commits_diff'] = df['text_commits_diff'].fillna('').astype(str).apply(lambda x: x[:200]) # FOR TEST
df['code_commits_diff'] = df['code_commits_diff'].fillna('').astype(str).apply(lambda x: x[:200]) # FOR TEST

# Tokenize text and code data
tokenized_text_train = tokenize_bert(df[df['year'] < 2022]['text_commits_diff'], tokenizer_text)
tokenized_text_test = tokenize_bert(df[df['year'] >= 2022]['text_commits_diff'], tokenizer_text)

tokenized_code_train = tokenize_bert(df[df['year'] < 2022]['code_commits_diff'], tokenizer_code)
tokenized_code_test = tokenize_bert(df[df['year'] >= 2022]['code_commits_diff'], tokenizer_code)

X_metadata_train = torch.tensor(X_metadata[df['year'] < 2022].values, dtype=torch.float32)
current_stars_train = torch.tensor(current_stars[df['year'] < 2022].values, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y[df['year'] < 2022].values, dtype=torch.float32)

X_metadata_test = torch.tensor(X_metadata[df['year'] >= 2022].values, dtype=torch.float32)
current_stars_test = torch.tensor(current_stars[df['year'] >= 2022].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y[df['year'] >= 2022].values, dtype=torch.float32)

# Create custom dataset to handle tokenized text and code
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, text, text_attention, code, code_attention, current_stars, labels):
        self.metadata = metadata
        self.text = text
        self.text_attention = text_attention
        self.code = code
        self.code_attention = code_attention
        self.current_stars = current_stars
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.metadata[idx],
            (self.text[idx], self.text_attention[idx]),
            (self.code[idx], self.code_attention[idx]),
            self.current_stars[idx],
            self.labels[idx]
        )

# Create datasets and dataloaders
train_dataset = CustomDataset(
    X_metadata_train,
    tokenized_text_train['input_ids'],
    tokenized_text_train['attention_mask'],
    tokenized_code_train['input_ids'],
    tokenized_code_train['attention_mask'],
    current_stars_train,
    y_train
)

test_dataset = CustomDataset(
    X_metadata_test,
    tokenized_text_test['input_ids'],
    tokenized_text_test['attention_mask'],
    tokenized_code_test['input_ids'],
    tokenized_code_test['attention_mask'],
    current_stars_test,
    y_test
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Define model, optimizer, and loss function
num_languages = len(df['main_language'].unique())
print(num_languages)
print(df['main_language'].unique())
num_years = len(df['year_encoded'].unique())
embedding_size = 128
num_numerical_features = len(metadata_columns) - 2
output_size = 1  # For regression

model = Starhub(num_years, num_languages, num_numerical_features, embedding_size, output_size, mode='regression')
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss()  # Mean Squared Error for regression

# Training and Testing
# Ensure that the train_model and test_model methods of Starhub class are implemented to handle the updated inputs
model.train_model(train_loader, optimizer, loss_function, epochs=10)
model.test_model(test_loader, loss_function)
