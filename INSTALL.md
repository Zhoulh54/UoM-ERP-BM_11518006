 Installation Guide

This project includes multiple time series forecasting models. To avoid dependency conflicts, we have created separate requirements files for different models.

## Requirements File Description

### 1. `requirements.txt` - General Models
Applicable to the following models:
- Linear Models: Ridge, Lasso, ElasticNet, PLS, etc.
- Neural Networks: MLP, LSTM
- Transformer Models: Autoformer, N-Beats
- Portfolio backtesting and evaluation

```bash
pip install -r requirements.txt
```

### 2. `requirements_chronos.txt` - Chronos-T5 Foundation Models
Applicable to Amazon's Chronos-T5 series models:
- Chronos-T5-tiny
- Chronos-T5-mini  
- Chronos-T5-small
- Chronos-T5-base
- Chronos-T5-large

```bash
pip install -r requirements_chronos.txt
```

### 3. `requirements_timesfm.txt` - TimesFM Foundation Models
Applicable to Google Research's TimesFM models:
- TimesFM-1.0
- TimesFM-2.0

```bash
pip install -r requirements_timesfm.txt
```

### 4. `requirements_uni2ts.txt` - Uni2TS Foundation Models  
Applicable to Uni2TS models:
- Uni2TS unified time series forecasting model

```bash
pip install -r requirements_uni2ts.txt
```

## Installation Method

### Using Pip Requirements

Install the required dependencies based on which models you plan to use:

```bash
# For general models (Linear, MLP, LSTM, Autoformer, N-Beats)
pip install -r requirements.txt

# For Chronos-T5 models
pip install -r requirements_chronos.txt

# For TimesFM models
pip install -r requirements_timesfm.txt

# For Uni2TS/Moirai models
pip install -r requirements_uni2ts.txt
```

**Note**: It's recommended to use virtual environments to avoid dependency conflicts between different model types.

### Special Note for Apple Silicon (M1/M2/M3)

For Apple Silicon Macs, some packages may require special installation steps:

```bash
# TensorFlow with Metal support
pip install tensorflow-macos tensorflow-metal
```

## Project Structure

ERP_Code/
├── 1_Data_Preprocessing/
├── 2_EDA/
├── 3_Out_of_Sample_Prediction/
│   ├── 1_Linear_Models/
│   ├── 2_Non_Linear_Models/         # MLP, LSTM, N-BEATS, Tree models
│   ├── 3_Transformer_Models/
│   └── 4_Foundation_Models/
├── 4_Portfolio_Backtesting/
├── 5_LLM_FineTuning/
├── requirements.txt
├── requirements_chronos.txt
├── requirements_timesfm.txt
└── requirements_uni2ts.txt
```

## Notes

1. **Version Compatibility**: All version numbers are based on tested environments in practice.
2. **GPU Support**: 
   - PyTorch: Automatically detects CUDA/MPS.
   - TensorFlow: Requires additional installation of tensorflow-metal (for Apple Silicon).
3. **Foundation Models**: Some foundation models may need to be installed from GitHub source code.
4. **Memory Requirements**: Foundation models require large memory, at least 16GB RAM is recommended.

