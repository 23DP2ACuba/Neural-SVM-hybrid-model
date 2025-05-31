# Neural-SVM-hybrid-model

## 📌 Features
- Historical crypto price data processing using `yfinance`
- Custom technical indicator engineering: VWAP, ATR, RSI, HL% change, and return features
- Neural Network with dropout and regularization
- Binary classification based on expected short-term return threshold
- SVM classifier trained on deep-learned features
- Visualization of training metrics and strategy performance

## 🔧 Configuration
Modify the following constants in the script to configure your experiment:

```python
SYMBOL = "BTC-USD" <- asset name
START = "2015-01-01" <- start date
END = "2025-04-17" <- end date
PERIOD = "1d" <- timeframe
LOOKBACK = 5
LOOKAHEAD = 5
expected_return = 2  <- threshold percentage for binary labeling
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
```
## 📈 Data Engineering Pipeline
- Fetch data using yfinance
- Feature engineering:
- Rolling close prices (LOOKBACK)
- HL% volatility
- VWAP, RSI, ATR

Labeling:

- Binary label overn = 1 if any price in next LOOKAHEAD days exceeds expected_return

- Robust scaling and train/val/test split (no shuffling)

## 🧠 Model Architecture
FeatureNet (PyTorch)

```python
Input → Linear(64) → ReLU → Dropout  
      → Linear(64) → ReLU → Dropout  
      → Linear(32) → ReLU  
      → Output(1) with BCEWithLogitsLoss
```
Class imbalance is addressed using pos_weight in the loss function.

🧪 Training & Evaluation
- Training loop includes:
- Epoch-wise tracking of loss, accuracy, precision, recall, and F1
- Early stopping logic (manual load of best model recommended)
- Visualizations for loss, accuracy, precision/recall, and F1 score

## 🔁 SVM on Learned Features
- After training, the neural network is used as a feature extractor. The output of the final hidden layer is passed into an SVM classifier (RBF kernel), which is trained on the training set and evaluated on the test set.

## 📊 Strategy Performance
- After prediction, a simple strategy return is calculated:

```python
strategy_return = predicted_signal * return_over_LOOKAHEAD
Cumulative returns are plotted to visualize profitability over time.
```
## 📁 Dependencies
- Install the required libraries using pip:

```bash
pip install torch scikit-learn yfinance matplotlib pandas numpy
```
## 📌 Output Example
- Training logs: Epoch-wise metrics
- Evaluation: Classification report, accuracy, precision, recall, F1
- Plots: Model training progress and cumulative strategy returns

## 📍 Notes
- The model is sensitive to the expected_return and LOOKAHEAD parameters. Tuning these can significantly impact performance.
- No shuffling is applied during train/test split due to the time series nature.
- RSI function (compute_rsi) must be defined or imported for full execution.
