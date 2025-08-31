# Exploring Advanced Financial Applications of Large Language Models

## Project Overview

This research project evaluates the effectiveness of large-scale Time-Series Foundation Models in financial forecasting applications. The study conducts a comprehensive comparison between modern foundation models (Chronos-T5, TimesFM, Moirai-1.1-R) and traditional forecasting approaches including linear models, ensemble methods, and deep learning architectures.

The research focuses on predicting daily excess returns for U.S. Consumer Discretionary stocks over a 24-year period (2000-2024). The evaluation framework encompasses both statistical performance metrics and practical economic outcomes through portfolio backtesting with realistic trading costs.

**Research Methodology:**
- **Model Comparison**: Foundation models vs. established benchmarks (Ridge, Lasso, Random Forest, LSTM, etc.)
- **Evaluation Framework**: Static out-of-sample testing and dynamic quarterly rebalancing strategies  
- **Economic Assessment**: Portfolio performance analysis with transaction cost sensitivity
- **Fine-tuning Analysis**: LoRA adaptation experiments for domain-specific optimization

**Key Research Contributions:**
- Comprehensive benchmark of foundation models in financial forecasting
- Analysis of domain transfer challenges for general-purpose time series models
- Investigation of the accuracy-utility trade-off in quantitative finance
- Practical guidance for applying large-scale models to financial prediction tasks

## Project Structure

```
ERP_Code/
├── 1_Data_Preprocessing/                    # Data cleaning and feature engineering
│   ├── Data_Cleaning_and_TrainTest_Datasets.ipynb
│   ├── 5_Factors_Plus_Momentum.csv
│   └── CRSP_ConsumerDiscretionary_*.csv
├── 2_EDA/                                  # Exploratory Data Analysis
│   └── Data_Analysis.ipynb
├── 3_Out_of_Sample_Prediction/            # Static out-of-sample forecasting
│   ├── 1_Linear_Models/                    # Linear and regularized models
│   ├── 2_Non_Linear_Models/                # MLP, LSTM, N-BEATS, Tree models
│   ├── 3_Transformer_Models/               # Autoformer (attention-based)
│   ├── 4_Foundation_Models/                # TSFMs (zero-shot)
│   │   ├── Chronos_T5/                     # Amazon Chronos-T5 variants
│   │   ├── TimesFM/                        # Google TimesFM
│   │   └── Moirai-1.1-R/                  # Salesforce Uni2TS/Moirai
│   └── 5_Unscaled_Dataset_Tests/          # Robustness tests
├── 4_Portfolio_Backtesting/                # Dynamic backtesting framework
│   ├── 1_Linear_Models/
│   ├── 2_Non_Linear_Models/                # Includes N-BEATS portfolio results
│   ├── 3_Transformer_Models/
│   ├── 4_Foundation_Models/
│   ├── 5_DM_Test/                          # Diebold-Mariano statistical tests
│   └── 6_Unscaled_Dataset_Tests/
├── 5_LLM_FineTuning/                       # LoRA fine-tuning experiments
│   ├── Chronos_T5/                         # Chronos fine-tuning
│   ├── TimesFM/                            # TimesFM fine-tuning
│   └── Moirai-1.1-R/                      # Moirai fine-tuning
├── requirements.txt                         # General models dependencies
├── requirements_chronos.txt                 # Chronos-specific dependencies
├── requirements_timesfm.txt                 # TimesFM-specific dependencies
├── requirements_uni2ts.txt                  # Uni2TS/Moirai dependencies
└── INSTALL.md                              # Detailed installation guide
```

## Methodology Overview

### Model Categories

| Category | Models | Rationale |
|----------|--------|-----------|
| **Linear & Regularized Models** | OLS, Ridge, Lasso, ENet, PCR, PLS | Statistical learning approaches with sparsity constraints and overfitting control mechanisms |
| **Tree-based Ensemble** | Random Forest, XGBoost | Non-parametric methods capable of modeling feature interactions and non-linear relationships |
| **Neural Networks** | MLP, LSTM, N-BEATS | Feedforward and recurrent architectures designed for learning temporal representations |
| **Attention-based Models** | Autoformer | Decomposition-enhanced transformers with learnable trend-seasonal attention patterns |
| **Foundation Models** | Chronos-T5, TimesFM, Moirai-1.1-R | Large-scale models pre-trained on diverse time series to assess transferability |

### Data & Features

- **Universe**: U.S. Consumer Discretionary sector (SIC 5000-5999, 7000-7999)
- **Period**: January 2000 - December 2024
- **Sample**: Top 50 stocks by market cap (fixed as of Dec 2015)
- **Features**: Rolling windows {5, 21, 252, 512} days of past excess returns
- **Target**: Next-day excess return
- **Split**: 2000-2015 (training), 2016-2024 (testing)

### Evaluation Framework

**Static Out-of-Sample Prediction:**
- Zero-based R², MSE, MAE
- Directional Accuracy (overall, up-days, down-days)

**Dynamic Portfolio Backtesting:**
- Quarterly expanding-window retraining
- Long-only, Short-only, Long-Short strategies
- Equal-weighted and Value-weighted portfolios
- Transaction costs: 5, 10, 20, 40 bps
- Risk-adjusted metrics: Sharpe ratio, Alpha, Information ratio

## Getting Started

### 1. Installation

Choose the appropriate requirements file based on your use case:

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

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### 2. Foundation Models Setup

Download the foundation models from their official repositories:

#### a. Chronos-T5 (Amazon Science)
```bash
git clone https://github.com/amazon-science/chronos-forecasting
```
**Repository**: https://github.com/amazon-science/chronos-forecasting

#### b. TimesFM (Google Research)
```bash
git clone https://github.com/google-research/timesfm
```
**Repository**: https://github.com/google-research/timesfm

#### c. Uni2TS/Moirai (Salesforce AI Research)
```bash
git clone https://github.com/SalesforceAIResearch/uni2ts
```
**Repository**: https://github.com/SalesforceAIResearch/uni2ts

### 3. Data Acquisition

**Important**: You must obtain CRSP data through legitimate WRDS subscription. Redistribution of CRSP data is prohibited.

#### WRDS Data Access:
1. Access [WRDS](https://wrds-www.wharton.upenn.edu/) (requires institutional subscription)
2. Navigate to CRSP database section (consult WRDS documentation for current menu structure)
3. Access daily stock/security data files
4. Apply the following filters:
   - **Date Range**: January 1, 2000 - December 31, 2024
   - **Exchanges**: NYSE, AMEX, NASDAQ 
   - **SIC Codes**: 5000-5999 (Retail Trade), 7000-7999 (Services)
5. Select required variables: PERMNO, PERMCO, TICKER, DATE, PRC, RET, VOL, SHROUT, SICCD, EXCHCD
6. Export data in CSV format and place in `1_Data_Preprocessing/` directory

#### Fama-French Factor Data (from WRDS):
Additional factor data is required for risk adjustment and portfolio analysis:

1. Access [WRDS](https://wrds-www.wharton.upenn.edu/) (same institutional subscription)
2. Navigate to Fama-French factor data section (consult WRDS documentation for current menu structure)
3. Select daily frequency factor data covering January 2000 - December 2024
4. Download the following factors:
   - Fama-French 5 Factors (daily)
   - Momentum Factor (daily)
   - Risk-free rate (daily)
5. Save as `5_Factors_Plus_Momentum.csv` in `1_Data_Preprocessing/` directory

**Required Factors:**
- Mkt-RF (Market excess return)
- SMB (Small minus Big)
- HML (High minus Low) 
- RMW (Robust minus Weak)
- CMA (Conservative minus Aggressive)
- Mom (Momentum)
- RF (Risk-free rate)


#### Sample Data Structures:

**CRSP Data:**
```csv
PERMNO,PERMCO,TICKER,DATE,PRC,RET,VOL,SHROUT,SICCD,EXCHCD
10001,7953,AMZN,2000-01-03,81.25,0.0234,5438400,359000,5961,3
10001,7953,AMZN,2000-01-04,78.94,-0.0284,3893200,359000,5961,3
...
```

**Factor Data:**
```csv
Date,Mkt-RF,SMB,HML,RMW,CMA,Mom,RF
20000103,0.0145,-0.0023,0.0067,0.0012,-0.0034,0.0089,0.0015
20000104,-0.0098,0.0034,-0.0045,0.0023,0.0056,-0.0023,0.0015
...
```

#### Data location

Place the processed dataset locally and point the code to it. Paths in this repo are examples only.

### 4. Workflow

1. **Data Preprocessing**: Start with `1_Data_Preprocessing/Data_Cleaning_and_TrainTest_Datasets.ipynb`
2. **Exploratory Analysis**: Run `2_EDA/Data_Analysis.ipynb`
3. **Static Forecasting**: Execute notebooks in `3_Out_of_Sample_Prediction/`
4. **Portfolio Backtesting**: Run notebooks in `4_Portfolio_Backtesting/`
5. **Fine-tuning**: Experiment with `5_LLM_FineTuning/`

## Environment and Reproducibility

### Experimental Hardware

| Component | Local Development | Foundation Model Training |
|-----------|-------------------|---------------------------|
| **Primary System** | MacBook Pro M4 Pro, 48GB RAM | University Computing Resources / Google Colab |
| **GPU** | Apple Silicon (MPS) | NVIDIA A100 (40GB VRAM) |
| **Use Case** | General models, data preprocessing, analysis | LoRA fine-tuning, large model inference |

### Tested Environment Matrix

| Component | Specification | Status |
|-----------|---------------|--------|
| **Python Versions** | 3.9, 3.10, 3.11, 3.12 | Verified |
| **Operating Systems** | macOS (Intel/Apple Silicon) | Verified |
| **Hardware** | CPU-only, CUDA GPUs, Apple Silicon (MPS) | Verified |
| **Memory** | 16GB+ RAM (32GB+ recommended for large foundation models) | Required |

### Reproducibility Settings

All experiments use **fixed random seeds (42)** across:
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- Python: `random.seed(42)`
- Optuna: `study = optuna.create_study(sampler=TPESampler(seed=42))`

### Minimum Requirements

- **Python**: 3.9+ (3.10 recommended for optimal compatibility)
- **RAM**: 16GB+ (32GB+ for foundation models)
- **Storage**: 10GB+ free space for datasets and model checkpoints
- **GPU**: Optional but recommended (CUDA or Apple Silicon MPS)
- **Internet**: Required for model downloads and WRDS access

## Acknowledgments

- **Professor Eghbal Rahimikia** for supervision and guidance throughout this research
- **University of Manchester** for providing research facilities and computational resources
- Amazon Science for Chronos-T5 foundation models
- Google Research for TimesFM foundation models
- Salesforce AI Research for Uni2TS/Moirai foundation models
- WRDS for providing access to financial data

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Academic Research Statement:**
This repository contains original research code and documentation developed for academic purposes at the University of Manchester.

