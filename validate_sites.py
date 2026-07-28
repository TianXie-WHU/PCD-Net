"""Site-based leave-one-out cross-validation on in situ LST samples."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from utils.data_loader import load_site_data
from utils.model import PCDNet
from utils.training import get_device, load_weights, train_model
from utils.validation import metric_row, predict_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/station_lst.csv")
    parser.add_argument(
        "--simulation-checkpoint-dir",
        default="models/simulation_finetuned",
    )
    parser.add_argument(
        "--predictions-output",
        default="results/site_loocv_predictions.csv",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/site_loocv_metrics.csv",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-sites", type=int, default=29)
    args = parser.parse_args()

    dataset, site_ids, source_frame = load_site_data(args.data)
    unique_sites = np.array(sorted(np.unique(site_ids)))
    if args.expected_sites and len(unique_sites) != args.expected_sites:
        raise ValueError(
            f"Expected {args.expected_sites} sites, found {len(unique_sites)}."
        )

    device = get_device()
    prediction_records = []
    metric_records = []

    with tempfile.TemporaryDirectory(prefix="pcdnet_site_loocv_") as temp_root:
        for fold_index, held_out_site in enumerate(unique_sites, start=1):
            train_indices = np.flatnonzero(site_ids != held_out_site)
            test_indices = np.flatnonzero(site_ids == held_out_site)
            print(
                f"Site fold {fold_index}/{len(unique_sites)} | "
                f"held out={held_out_site} | "
                f"train={len(train_indices)}, test={len(test_indices)}"
            )

            # Every fold starts from the same simulation-domain checkpoint.
            model = load_weights(
                PCDNet(),
                args.simulation_checkpoint_dir,
                device,
            )
            train_model(
                model,
                Subset(dataset, train_indices.tolist()),
                Path(temp_root) / f"{fold_index:02d}_{held_out_site}",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                seed=args.seed + fold_index,
            )

            observations, predictions = predict_dataset(
                model,
                Subset(dataset, test_indices.tolist()),
                device,
                batch_size=args.batch_size,
            )
            metric_records.append(
                metric_row(held_out_site, observations, predictions)
            )

            fold_frame = source_frame.iloc[test_indices].copy()
            fold_frame.insert(0, "source_index", test_indices)
            fold_frame.insert(0, "fold", fold_index)
            fold_frame["predicted_lst"] = predictions
            fold_frame["error"] = predictions - observations
            prediction_records.append(fold_frame)

    predictions_frame = pd.concat(prediction_records, ignore_index=True)
    predictions_frame = predictions_frame.sort_values("source_index")
    overall = metric_row(
        "overall",
        predictions_frame["lst"].to_numpy(),
        predictions_frame["predicted_lst"].to_numpy(),
    )
    metric_records.append(overall)

    prediction_path = Path(args.predictions_output)
    metric_path = Path(args.metrics_output)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(prediction_path, index=False)
    pd.DataFrame(metric_records).to_csv(metric_path, index=False)

    print(f"Predictions saved to {prediction_path}")
    print(f"Metrics saved to {metric_path}")
    print(
        "Overall | "
        f"RMSE={overall['rmse']:.4f}, MAE={overall['mae']:.4f}, "
        f"Bias={overall['bias']:.4f}, R2={overall['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
