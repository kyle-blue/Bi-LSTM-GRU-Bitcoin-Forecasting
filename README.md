# Bi-LSTM-GRU-Bitcoin-Forecasting

Application allows the creation and testing of (bidirectional or unidirectional) GRU and LSTM models which forecast future price direction of Bitcoin. It is simple to switch between testing classification or regression models (see app.py)

## Datasets
- Datasets were downloaded from Kaggle [here](https://www.kaggle.com/jorijnsmit/binance-full-history)
- These should be placed in ./dataset/crypto/ folder


## Prerequisites

- Have python 3.9 installed
- Have VS Build tools 2014+ and Visual C++ build tools (with windows sdk)
- If using windows, do not run this in WSL, you may run into GPU issues