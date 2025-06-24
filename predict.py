#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple prediction script.

This script loads a raw CSV file, processes and scales it using pre-saved
scalers, then performs a prediction with a saved model.
"""

import os
import argparse
import pandas as pd
import torch
import numpy as np

from data.finance.data_processor import StockDataProcessor
from model.transformer import StockPricePredictor
from log.logger import get_logger


logger = get_logger(__name__, log_file="predict.log")


def load_and_prepare(csv_path: str, ticker: str, processor: StockDataProcessor,
                     sequence_length: int, prediction_horizon: int):
    """Load raw CSV and convert it into scaled sequences."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure ticker column exists for compatibility
    if 'ticker' not in df.columns:
        df['ticker'] = ticker

    # Clean and add features
    df = processor.clean_stock_data(df)
    df = processor.add_features(df)

    target_col = f"future_return_{prediction_horizon}d"
    if target_col not in df.columns:
        logger.warning(f"{target_col} not found, proceeding without target scaling")
        target_col = None

    # Determine feature columns
    feature_cols = [c for c in df.columns
                    if c not in ['ticker', 'date', target_col] and
                    pd.api.types.is_numeric_dtype(df[c])]

    # Scale using loaded scalers
    scaled = df.copy()
    processor.load_scalers(ticker)
    if 'features' in processor.scalers:
        scaled[feature_cols] = processor.scalers['features'].transform(df[feature_cols])
    if target_col and 'target' in processor.scalers:
        scaled[target_col] = processor.scalers['target'].transform(
            df[target_col].values.reshape(-1, 1)).flatten()

    sequences = processor.create_sequence_datasets(
        {'train': scaled},
        feature_columns=feature_cols,
        target_column=target_col,
        sequence_length=sequence_length
    )

    X = sequences['train'][0]
    if X.size == 0:
        raise ValueError("No sequences could be created from the provided data")
    return X[-1:], processor.scalers.get('target'), target_col


def predict_from_csv(csv_path: str, ticker: str, model_path: str,
                      sequence_length: int = 32, prediction_horizon: int = 2,
                      processed_dir: str = "data/finance/processed"):
    """Predict future return using saved model and scalers."""
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    processor = StockDataProcessor(
        raw_data_dir=os.path.dirname(csv_path),
        processed_data_dir=processed_dir,
        scaler_type="robust"
    )

    features, target_scaler, target_col = load_and_prepare(
        csv_path, ticker, processor, sequence_length, prediction_horizon)

    predictor = StockPricePredictor.load(model_path, device=device)

    tensor = torch.tensor(features, dtype=torch.float32).to(device)
    output = predictor.predict(tensor)
    pred = output[-1].cpu().numpy()

    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    return float(pred[0])


def main():
    parser = argparse.ArgumentParser(description="Predict using saved model")
    parser.add_argument("csv", help="Path to raw CSV file")
    parser.add_argument("ticker", help="Ticker symbol (used to load scalers)")
    parser.add_argument("--model", default="models/SpaceExploreAI_best.pt",
                        help="Path to saved model")
    parser.add_argument("--sequence-length", type=int, default=32,
                        help="Sequence length")
    parser.add_argument("--horizon", type=int, default=2,
                        help="Prediction horizon used during training")
    parser.add_argument("--processed-dir", default="data/finance/processed",
                        help="Directory where scalers are stored")
    args = parser.parse_args()

    try:
        prediction = predict_from_csv(
            args.csv,
            args.ticker,
            args.model,
            sequence_length=args.sequence_length,
            prediction_horizon=args.horizon,
            processed_dir=args.processed_dir
        )
        print(f"Prediction for {args.ticker}: {prediction:.6f}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
