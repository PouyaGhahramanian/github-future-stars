import torch
import torch.nn as nn
from transformers import BertModel

class MetaDataEmbedding(nn.Module):
    def __init__(self, num_years, num_languages, num_numerical_features, embedding_size):
        super().__init__()
        self.year_embedding = nn.Embedding(num_years, embedding_size)
        self.language_embedding = nn.Embedding(num_languages, embedding_size)
        self.fc_numerical = nn.Linear(num_numerical_features, embedding_size)
        self.fc_combined = nn.Linear(3 * embedding_size, embedding_size)

    def forward(self, year_index, language_index, numerical_features):
        year_embedded = self.year_embedding(year_index)
        language_embedded = self.language_embedding(language_index)
        numerical_embedded = torch.relu(self.fc_numerical(numerical_features))
        combined = torch.cat([year_embedded, language_embedded, numerical_embedded], dim=1)
        combined = torch.relu(self.fc_combined(combined))
        return combined

class Starhub(nn.Module):
    def __init__(self, num_years, num_languages, meta_data_input_size, meta_data_embedding_size, output_size, mode='regression'):
        super().__init__()
        self.meta_data_embedding = MetaDataEmbedding(num_years, num_languages, meta_data_input_size, meta_data_embedding_size)
        self.text_embedding_model = BertModel.from_pretrained('bert-base-uncased')
        self.code_embedding_model = BertModel.from_pretrained('microsoft/codebert-base')
        self.mode = mode
        combined_embedding_size = meta_data_embedding_size + self.text_embedding_model.config.hidden_size + self.code_embedding_model.config.hidden_size + 1 # For current_stars feature
        self.classifier = nn.Linear(combined_embedding_size, output_size)

    def forward(self, meta_data, text, code, current_stars):
        year_index = meta_data[:, 0].long()
        language_index = meta_data[:, 1].long()
        numerical_features = meta_data[:, 2:]
        meta_data_embedded = self.meta_data_embedding(year_index, language_index, numerical_features)
        text_ids, text_attention = text
        text_embedded = self.text_embedding_model(text_ids, attention_mask=text_attention)[1]
        code_ids, code_attention = code
        code_embedded = self.code_embedding_model(code_ids, attention_mask=code_attention)[1]
        combined = torch.cat([meta_data_embedded, text_embedded, code_embedded, current_stars], dim=1)
        output = self.classifier(combined)
        if self.mode == 'classification':
            return torch.sigmoid(output)
        else:
            return output

    def train_model(self, train_loader, optimizer, loss_function, epochs):
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars, targets = batch
                optimizer.zero_grad()
                outputs = self(metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    def test_model(self, test_loader, loss_function=None):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars, targets = batch
                outputs = self(metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars)
                if loss_function:
                    loss = loss_function(outputs, targets)
                    total_loss += loss.item()

        if loss_function:
            print(f"Test Loss: {total_loss/len(test_loader)}")

class Featurehub(nn.Module):
    def __init__(self, num_years, num_languages, meta_data_input_size, meta_data_embedding_size, output_size, mode='regression'):
        super().__init__()
        self.meta_data_embedding = MetaDataEmbedding(num_years, num_languages, meta_data_input_size, meta_data_embedding_size)

        self.mode = mode
        combined_embedding_size = meta_data_embedding_size + 1
        self.classifier = nn.Linear(combined_embedding_size, output_size)

    def forward(self, meta_data, text, code, current_stars):
        year_index = meta_data[:, 0].long()
        language_index = meta_data[:, 1].long()
        numerical_features = meta_data[:, 2:]
        meta_data_embedded = self.meta_data_embedding(year_index, language_index, numerical_features)
        combined = torch.cat([meta_data_embedded, current_stars], dim=1)
        output = self.classifier(combined)
        if self.mode == 'classification':
            return torch.sigmoid(output)
        else:
            return output

    def train_model(self, train_loader, optimizer, loss_function, epochs):
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars, targets = batch
                optimizer.zero_grad()
                outputs = self(metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    def test_model(self, test_loader, loss_function=None):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars, targets = batch
                outputs = self(metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars)
                if loss_function:
                    loss = loss_function(outputs, targets)
                    total_loss += loss.item()

        if loss_function:
            print(f"Test Loss: {total_loss/len(test_loader)}")