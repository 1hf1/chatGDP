import pandas as pd
from fredapi import Fred
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import joblib
import torch
series_map = {
    "GDP and Economic Growth": {
        'GDPC1': 'Real_GDP',
        'A939RX0Q048SBEA': 'Real_Disposable_Income',
        'PCE': 'Personal_Consumption',
    },
    "Labor Market": {
        'UNRATE': 'Unemployment_Rate',
        'PAYEMS': 'Nonfarm_Payrolls',
        'CES0500000003': 'Avg_Hourly_Earnings',
        'LNS11300060': 'U6_Unemployment',
        'JTSJOL': 'Job_Openings',
        'LNS12300060': 'Labor_Force_Participation',
    },
    "Inflation and Prices": {
        'CPIAUCSL': 'CPI_All_Items',
        'CPILFESL': 'Core_CPI',
        'PPIACO': 'Producer_Price_Index',
        'PCEPILFE': 'Core_PCE_Price_Index',
    },
    "Monetary Policy and Finance": {
        'FEDFUNDS': 'Federal_Funds_Rate',
        'DGS10': '10Y_Treasury_Yield',
        'M2SL': 'M2_Money_Stock',
        'WALCL': 'Fed_Balance_Sheet',
        'DTB3': '3M_Treasury_Yield',
        'MORTGAGE30US': '30Y_Mortgage_Rate',
        'T10YIE': '10Y_Inflation_Expect',
        'USD3MTD156N': '3M_LIBOR',
    },
    "Financial Markets": {
        'SP500': 'S&P_500_Index',
        'VIXCLS': 'VIX_Index',
        'DTWEXB': 'Dollar_Index',
        'DEXUSEU': 'USD_EUR_Rate',
        'GOLDAMGBD228NLBM': 'Gold_Price',
    },
    "Production and Business Activity": {
        'INDPRO': 'Industrial_Production',
        'TCU': 'Capacity_Utilization',
        'IPB51100N': 'Business_Equipment_Prod',
        'IPG211S': 'Energy_Production',
        'IPCONGD': 'Consumer_Goods_Prod',
        'AMTMNO': 'New_Manufacturing_Orders',
    },
    "Consumer Behavior": {
        'UMCSENT': 'Consumer_Sentiment',
        'RSAFS': 'Retail_Sales',
        'PSAVERT': 'Personal_Saving_Rate',
        'TOTALSL': 'Consumer_Credit',
        'RRSFS': 'Retail_Sales_Food_Services',
    },
    "Housing Market": {
        'HOUST': 'Housing_Starts',
        'CSUSHPINSA': 'Case_Shiller_Home_Price',
        'MSPUS': 'Median_Home_Price',
        'HSN1F': 'New_Home_Sales',
        'BPPRIV': 'Building_Permits',
    },
    "International Trade": {
        'BOPGSTB': 'Trade_Balance',
        'XTEXVA01USM667S': 'Exports',
        'XTIMVA01USM667S': 'Imports',
    },
    "Banking and Credit": {
        'REALLN': 'Commercial_Loans',
        'TOTCI': 'C&I_Loans',
        'EXCSRESNW': 'Excess_Reserves',
    },
    "Energy and Commodities": {
        'DCOILWTICO': 'Crude_Oil_Price',
    },
    "Regional Indicators": {
        'NYUR': 'NY_Unemployment',
        'CASACBW027SBOG': 'CA_Commercial_Loans',
    },
    "Economic Indices": {
        'USALOLITONOSTSAM': 'Leading_Index',
        'STLFSI4': 'Financial_Stress_Index',
        'WABSI': 'Bank_Stress_Index',
    }
}

class EconomicDataset(Dataset):
    def __init__(self, data):
        """
        Wraps a numpy array (or pandas DataFrame) into a PyTorch Dataset.
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def clean_quarterly_data(df, col_thresh=0.3):
    """
    Cleans and prepares quarterly economic data by:
      1. Dropping columns with > col_thresh fraction of missing values.
      2. Forward- and backward-filling remaining missing values.
      3. Dropping rows still containing missing values.
      4. Dropping non-numeric columns (e.g., dates/strings).
    """
    clean_df = df.copy()

    # Drop columns exceeding col_thresh fraction of missing
    drop_cols = [col for col in clean_df.columns if clean_df[col].isna().mean() > col_thresh]
    clean_df.drop(columns=drop_cols, inplace=True)

    # Forward fill, then backward fill
    clean_df = clean_df.ffill().bfill()

    clean_df.dropna(how='any', inplace=True)

    # Drop non-numeric columns (dates, strings, etc.)
    #    This prevents "could not convert string to float" errors.
    numeric_cols = clean_df.select_dtypes(include=["number"]).columns
    clean_df = clean_df[numeric_cols]

    print(f"Cleaned dataset shape: {clean_df.shape}")
    return clean_df

def prepare_data(csv_path, test_size=0.2, batch_size=32):
    # Load raw data
    df = pd.read_csv(csv_path)
    print(f"Original dataset shape: {df.shape}")

    # Store feature names
    feature_names = df.columns.tolist()

    # Clean data
    cleaned_df = clean_quarterly_data(df, col_thresh=0.3)

    # Scale numeric data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_df.values)

    # Split train/test
    split_idx = int((1 - test_size) * len(scaled_data))
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx:]

    # Create Datasets
    train_dataset = EconomicDataset(train_data)
    test_dataset = EconomicDataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training dataset shape: {train_data.shape}")
    print(f"Testing dataset shape: {test_data.shape}")

    return train_loader, test_loader, scaler, feature_names
# Initialize FRED API (get free key: https://fred.stlouisfed.org/docs/api/api_key.html)
fred = Fred(api_key='b671e6c86011c87f2f5a87af22304559')


def collect_data(series_map):
    """Collect data for all series in SERIES_MAP using the FRED API."""
    fred = Fred(api_key='b671e6c86011c87f2f5a87af22304559')
    df = pd.DataFrame()

    params = {
        'frequency': 'q',
        'aggregation_method': 'avg'
    }

    for category, series_dict in series_map.items():
        for series_id, series_name in series_dict.items():
            try:
                series = fred.get_series(series_id, **params)
                df[series_name] = series
                print(f"Added {series_name} ({series_id})")
                time.sleep(0.2)  # FRED rate limit: 120 requests/minute
            except Exception as e:
                print(f"Failed to fetch {series_name} ({series_id}): {str(e)}")

    return df