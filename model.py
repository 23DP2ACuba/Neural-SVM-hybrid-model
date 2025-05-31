import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------ config --------------
SYMBOL = "BTC-USD"
START = "2015-01-01"
END = "2025-04-17"
PERIOD = "1d"
BATCH_SIZE = 32
LOOKBACK = 5
LOOKAHEAD = 5
expected_return = 2
LEARNING_RATE = 0.001
EPOCHS = 100
DROPOUT = 0.2
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# ----------------------------------


data = yf.Ticker(SYMBOL).history(period=PERIOD, start=START, end=END)[["Open", "High", "Low", "Close", "Volume"]]
data.columns = data.columns.str.lower()

def add_lookback(data):
    for i in range(LOOKBACK):
        data[f"close{i+1}"] = data['close'].shift(i+1)
    return data

def add_pct_chng(data):
  data["HL_pct"] = (data.high - data.low) / data.close * 100
  return data

def return_over_period(data):
    data["return"] = (data["close"].shift(-LOOKAHEAD) - data["close"]) / data["close"] * 100
    return data

def over_npct_span(data):
    over_threshold = []
    for i in range(1, 1 + LOOKAHEAD):
        future_return = (data["close"].shift(-i) - data["close"]) / data["close"] * 100
        over_threshold.append(future_return > expected_return)
    
    data["overn"] = np.any(over_threshold, axis=0).astype(int)
    return data

def over_npct(data):
    data["overn"] = np.where(data["return"] > 2.0, 1, 0)
    return data


def VWAP(data):
    data["vwap"] = (((data['high'] + data['close'] + data['low']) / 3) * data['volume']).cumsum() / data['volume'].cumsum()
    return data

def compute_atr(data, period=14):
    high = data['high']
    low = data['low']
    close = data['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low), 
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    data['atr'] = atr
    return data



data = add_lookback(data)
data = return_over_period(data)
data = VWAP(data)
data = compute_atr(data)
data = compute_rsi(data)
data = return_over_period(data)
data = add_pct_chng(data)
data = over_npct(data)
data.dropna(inplace=True)

feature_columns = data.columns.difference(["overn", "return", "high", "low"])
x = data[feature_columns].to_numpy()
y = data["overn"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False)

scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

class TSDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

class FeatureNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeatureNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        x = self.output(x)
        return x

input_size = len(feature_columns)  
model = FeatureNet(input_size=input_size, hidden_size=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
class_weights = torch.tensor([1.0, len(y_train)/(2*sum(y_train))], dtype=torch.float).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

train_dataset = TSDataset(x_train_scaled, y_train)
val_dataset = TSDataset(x_val_scaled, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

def train_epoch():
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (predicted == batch_y).sum().item()
        total_samples += batch_y.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def evaluate_val():
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

best_val_loss = float('inf')
patience_counter = 0
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_val()
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1s.append(val_f1)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, "
          f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
    
model.load_state_dict(torch.load('best_model.pt'))

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Neural Network Loss History')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Neural Network Accuracy History')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1, len(val_precisions)+1), val_precisions, label='Validation Precision')
plt.plot(range(1, len(val_recalls)+1), val_recalls, label='Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Neural Network Precision and Recall')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, len(val_f1s)+1), val_f1s, label='Validation F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Neural Network F1 History')
plt.legend()

plt.tight_layout()
plt.show()

model.eval()
with torch.no_grad():
    features_train = model(torch.tensor(x_train_scaled, dtype=torch.float).to(device)).cpu().numpy()
    features_test = model(torch.tensor(x_test_scaled, dtype=torch.float).to(device)).cpu().numpy()

svc = SVC(kernel='rbf', C=1.0, class_weight='balanced')
svc.fit(features_train, y_train)

y_pred = svc.predict(features_test)
svm_accuracy = accuracy_score(y_test, y_pred)
print("\nSVM Evaluation:")
print(f"Test Accuracy: {svm_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

test_data = data.iloc[-len(y_test):].copy()
test_data['predicted'] = y_pred
test_data['strategy_return'] = test_data['return'].shift(-LOOKAHEAD) * test_data['predicted']
cumulative_return = (1 + test_data['strategy_return'] / 100).cumprod() - 1
plt.figure(figsize=(10, 5))
plt.plot(cumulative_return)
plt.title('Cumulative Strategy Return')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.show()
