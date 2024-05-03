import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset

class DNASeqClassifier(nn.Module):
    def __init__(self, sequence_length):
        super(DNASeqClassifier, self).__init__()
        self.conv1 = nn.Conv1d(4, 300, kernel_size=19, padding=9)
        self.bn1 = nn.BatchNorm1d(300)
        self.conv2 = nn.Conv1d(300, 200, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(200)
        self.conv3 = nn.Conv1d(200, 200, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(200)
        self.pool = nn.MaxPool1d(3, stride=3)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout_fc = nn.Dropout(0.5)
        # Adjust the input features based on the actual output size before the fully connected layer
        self.fc1 = nn.Linear(5800, 1000)  # Adjust this to match the actual output size
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.dropout1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(self.dropout2(x))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(self.dropout3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(self.dropout_fc(x)))
        x = self.relu(self.fc2(self.dropout_fc(x)))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for sequences, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * sequences.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def validate_model(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    progress_bar = tqdm(test_loader, desc="Validating")
    
    with torch.no_grad():
        for sequences, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)  # Move data to GPU
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item() * sequences.size(0)
            
            # Assuming your model outputs logits and you use a binary classification
            # Convert outputs to binary predictions (0 or 1)
            predicted_labels = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Calculate current accuracy
            current_accuracy = (correct_predictions / total_predictions) * 100
            progress_bar.set_postfix(loss=loss.item(), accuracy=f"{current_accuracy:.2f}%")
    
    total_loss = running_loss / len(test_loader.dataset)
    total_accuracy = (correct_predictions / total_predictions) * 100
    print(f'Validation Loss: {total_loss:.4f}, Accuracy: {total_accuracy:.2f}%')

class DNADataset(Dataset):
    def __init__(self, dataframe):
        # Use a list comprehension to prepare the sequences, removing the 'N' channel
        self.sequences = [torch.tensor(x[:, :4].transpose((1, 0)), dtype=torch.float32) for x in dataframe['one_hot_data']]
        self.labels = torch.tensor(dataframe['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Each item is a tuple of (sequence, label)
        return self.sequences[idx], self.labels[idx]