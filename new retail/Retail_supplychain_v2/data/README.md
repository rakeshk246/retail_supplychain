# Data Directory

## M5 Forecasting Competition Dataset (Walmart)

To use real retail data, download the M5 dataset from Kaggle:

### Option 1: Kaggle CLI
```bash
pip install kaggle
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d .
```

### Option 2: Manual Download
1. Go to https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
2. Download `sales_train_evaluation.csv`
3. Place it in this `data/` directory

### Required File
- `sales_train_evaluation.csv` — 5 years of daily sales data across 3,000+ products

### Fallback
If the M5 dataset is not available, the system will automatically use enhanced synthetic data with realistic seasonality patterns.
