import torch
import torch.nn as nn
from transformers import BertModel

class MetaDataEmbedding(nn.Module):
    def __init__(self, num_years, num_languages, num_numerical_features, embedding_size):
        super().__init__()
        self.year_embedding = nn.Embedding(num_years, embedding_size)
        self.language_embedding = nn.Embedding(num_languages, embedding_size)  # +1 for unknown languages
        self.fc_numerical = nn.Linear(num_numerical_features, embedding_size)
        self.fc_combined = nn.Linear(3 * embedding_size, embedding_size)

    def forward(self, year_index, language_index, numerical_features):
        # Assuming language_index is already converted to an integer index
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
        # print(year_index)
        language_index = meta_data[:, 1].long()  # Convert to long as language_index needs to be a long tensor for nn.Embedding
        # print(language_index)
        # print(meta_data)
        numerical_features = meta_data[:, 2:]  # Rest of the columns after the language index are numerical features
        meta_data_embedded = self.meta_data_embedding(year_index, language_index, numerical_features)
        text_ids, text_attention = text
        text_embedded = self.text_embedding_model(text_ids, attention_mask=text_attention)[1]
        code_ids, code_attention = code
        code_embedded = self.code_embedding_model(code_ids, attention_mask=code_attention)[1]

        print(f"meta_data_embedded shape: {meta_data_embedded.shape}")
        print(f"text_embedded shape: {text_embedded.shape}")
        print(f"code_embedded shape: {code_embedded.shape}")
        print(f"current_stars shape: {current_stars.unsqueeze(1).shape}")

        combined = torch.cat([meta_data_embedded, text_embedded, code_embedded, current_stars.unsqueeze(1)], dim=1)
        output = self.classifier(combined)
        if self.mode == 'classification':
            return torch.sigmoid(output)
        else:
            return output

    def train_model(self, train_loader, optimizer, loss_function, epochs):
        self.train()  # set the model to training mode
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                # Unpack the batch
                metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars, targets = batch
                optimizer.zero_grad()  # clear previous gradients

                # Pass inputs along with their attention masks
                outputs = self(metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars)
                loss = loss_function(outputs, targets)
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    def test_model(self, test_loader, loss_function=None):
        self.eval()  # set the model to evaluation mode
        total_loss = 0
        with torch.no_grad():  # no gradient computation
            for batch in test_loader:
                # Unpack the batch
                metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars, targets = batch

                # Pass inputs along with their attention masks
                outputs = self(metadata, (text_ids, text_mask), (code_ids, code_mask), current_stars)
                if loss_function:
                    loss = loss_function(outputs, targets)
                    total_loss += loss.item()

        if loss_function:
            print(f"Test Loss: {total_loss/len(test_loader)}")

if __name__ == '__main__':
    meta_data_input_size = 10
    meta_data_embedding_size = 128
    num_languages = 20
    output_size = 1  # For regression, it's 1; for classification, it could be the number of classes

    model = Starhub(num_languages, meta_data_input_size, meta_data_embedding_size, output_size, mode='regression')
