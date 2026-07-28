"""Evaluate a trained model or generate LST predictions."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import load_prediction_data
from utils.metrics import calculate_metrics
from utils.model import PCDNet
from utils.training import get_device, load_weights, unpack_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample.csv")
    parser.add_argument("--checkpoint-dir", default="models/pretrained")
    parser.add_argument("--output", default="results/predictions.csv")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    frame, inputs, observed = load_prediction_data(args.data)
    device = get_device()
    model = load_weights(PCDNet(), args.checkpoint_dir, device).to(device)
    model.eval()

    predictions = []
    loader = DataLoader(TensorDataset(inputs), batch_size=args.batch_size)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            predictions.extend(model(*unpack_inputs(batch)).cpu().numpy())

    output = frame.copy()
    output["predicted_lst"] = predictions
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    if observed is not None:
        for name, value in calculate_metrics(observed, predictions).items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
