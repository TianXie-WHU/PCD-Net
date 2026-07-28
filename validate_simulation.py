"""Ten-fold cross-validation on the radiative-transfer simulation dataset."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from utils.data_loader import load_component_data, load_lst_data
from utils.model import PCDNet
from utils.training import get_device, set_seed, train_model
from utils.validation import make_kfold_indices, metric_row, predict_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component-data",
        default="data/simulation_components.csv",
    )
    parser.add_argument("--lst-data", default="data/simulation_lst.csv")
    parser.add_argument(
        "--predictions-output",
        default="results/simulation_10fold_predictions.csv",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/simulation_10fold_metrics.csv",
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--component-epochs", type=int, default=200)
    parser.add_argument("--simulation-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--component-learning-rate", type=float, default=1e-5)
    parser.add_argument("--residual-learning-rate", type=float, default=1e-3)
    parser.add_argument("--simulation-learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    component_dataset = load_component_data(args.component_data)
    lst_dataset = load_lst_data(args.lst_data)
    if len(component_dataset) != len(lst_dataset):
        raise ValueError("Component and LST simulation files have different lengths.")
    if not np.allclose(
        component_dataset.tensors[0].numpy(),
        lst_dataset.tensors[0].numpy(),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("Component and LST simulation files are not row-aligned.")

    source_frame = pd.read_csv(args.lst_data)
    device = get_device()
    prediction_records = []
    metric_records = []

    with tempfile.TemporaryDirectory(prefix="pcdnet_simulation_cv_") as temp_root:
        for fold_index, (train_indices, test_indices) in enumerate(
            make_kfold_indices(len(lst_dataset), args.folds, args.seed),
            start=1,
        ):
            print(
                f"Simulation fold {fold_index}/{args.folds}: "
                f"train={len(train_indices)}, test={len(test_indices)}"
            )
            set_seed(args.seed + fold_index)
            model = PCDNet()
            fold_root = Path(temp_root) / f"fold_{fold_index:02d}"

            train_model(
                model,
                Subset(component_dataset, train_indices.tolist()),
                fold_root / "components",
                component_training=True,
                epochs=args.component_epochs,
                batch_size=args.batch_size,
                learning_rate=args.component_learning_rate,
                residual_learning_rate=args.residual_learning_rate,
                patience=args.patience,
                seed=args.seed + fold_index,
            )
            train_model(
                model,
                Subset(lst_dataset, train_indices.tolist()),
                fold_root / "simulation_finetuned",
                epochs=args.simulation_epochs,
                batch_size=args.batch_size,
                learning_rate=args.simulation_learning_rate,
                patience=args.patience,
                seed=args.seed + fold_index,
            )

            observations, predictions = predict_dataset(
                model,
                Subset(lst_dataset, test_indices.tolist()),
                device,
                batch_size=args.batch_size,
            )
            metric_records.append(
                metric_row(f"fold_{fold_index}", observations, predictions)
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
