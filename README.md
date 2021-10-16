# Bi-LSTM-GRU-Bitcoin-Forecasting

Application allows the creation and testing of (bidirectional or unidirectional) GRU and LSTM models which forecast future price direction of Bitcoin. It is simple to switch between testing classification or regression models (see app.py)

## Datasets
- Datasets were downloaded from Kaggle [here](https://www.kaggle.com/jorijnsmit/binance-full-history)
- These should be placed in ./dataset/crypto/ folder


## Prerequisites

- Have python 3.9 installed
- Have VS Build tools 2014+ and Visual C++ build tools (with windows sdk)
- If using windows, do not run this in WSL, you may run into GPU issues

### Install in this order:

- CUDA toolkit: https://developer.nvidia.com/cuda-toolkit-archive
- CudNN: https://developer.nvidia.com/cudnn (install instructions: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
- (optional but improves gpu latency) TensorRT: https://developer.nvidia.com/tensorrt-getting-started (install instructions: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip)

More up to date instructions may live here: https://www.tensorflow.org/install/gpu