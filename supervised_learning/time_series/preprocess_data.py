#!/usr/bin/env python3
""" Script to preprocess data """
import pandas as pd

def preprocess_data(coinbase_path, bitstamp_path, n=5):
    # Load the datasets
    df_coinbase = pd.read_csv(coinbase_path)
    df_bitstamp = pd.read_csv(bitstamp_path)

    # Convert Unix time to datetime
    df_coinbase['Timestamp'] = pd.to_datetime(df_coinbase['Timestamp'], unit='s')
    df_bitstamp['Timestamp'] = pd.to_datetime(df_bitstamp['Timestamp'], unit='s')

    # Merge datasets on Timestamp
    df = pd.merge(df_coinbase, df_bitstamp, on='Timestamp', suffixes=('_coinbase', '_bitstamp'))

    # Replace NaN values in the Close and Weighted_Price columns
    df['Close'] = df['Close_bitstamp'].combine_first(df['Close_coinbase'])
    df['Weighted_Price'] = df['Weighted_Price_bitstamp'].combine_first(df['Weighted_Price_coinbase'])

    # Keep only relevant columns
    df = df[['Timestamp', 'Close', 'Weighted_Price']]
    
    # Set 'Timestamp' as the index and resample to hourly data
    df.set_index('Timestamp', inplace=True)
    df = df.resample('h').mean()

    print(df.head())

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]
    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    train_df = train_df.diff().dropna()
    val_df = val_df.diff().dropna()
    test_df = test_df.diff().dropna()
    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)
    return train_df, val_df, test_df, train_mean, train_std
