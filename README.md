# PCD-Net

PCD-Net - Parallel Component Decoupled Neural Network is a mechanism-coupled neural network for medium- to high-resolution land surface temperature (LST) retrieval. It uses the generalized split-window (SW) equation as its physical backbone:

```text
LST = T10 + a0 + a1 × (T10 - T11) + a2 × (T10 - T11)² + a3
```

Four parallel subnetworks (SWBP1-SWBP4) learn the dynamic components `a0`, `a1`, `a2`, and `a3`. Each subnetwork uses atmospheric water vapor, mean emissivity, and emissivity difference as inputs.

The model is trained in three stages:

1. The four component subnetworks are initialized using radiative-transfer simulation labels.

2. The complete PCD-Net is jointly optimized using simulated LST samples.

3. The simulation-trained model is fine-tuned using in situ LST observations.

For methodological details, please refer to the [paper](https://arxiv.org/abs/2509.04991).

## Project structure

```text
PCD-Net/
├── data/
│   └── sample.csv              # Ten records showing the input format
├── models/
│   └── pretrained/
│       ├── SWBP1.pth           # a0 component network
│       ├── SWBP2.pth           # a1 component network
│       ├── SWBP3.pth           # a2 component network
│       └── SWBP4.pth           # coupling-residual network
├── utils/
│   ├── data_loader.py          # Data loading and checking
│   ├── metrics.py              # Evaluation metrics
│   ├── model.py                # PCD-Net architecture
│   ├── training.py             # Shared training functions
│   └── validation.py           # Cross-validation utilities
├── train_components.py         # Component-network initialization
├── train_simulation.py         # Simulation-domain joint optimization
├── train_station.py            # Fine-tuning with station observations
├── validate_simulation.py      # Simulation-data 10-fold cross-validation
├── validate_sites.py           # 29-site leave-one-site-out validation
├── test.py                     # Prediction and evaluation with trained models
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducibility statement

### System requirements

* Hardware: NVIDIA GPU recommended; CPU execution is also supported.

* Operating system: Linux, Windows, or macOS (64-bit).

* Python: version 3.9 or later.

### Dependency installation

```bash
conda create -n pcdnet python=3.9
conda activate pcdnet
pip install -r requirements.txt
```

The core dependencies are PyTorch, NumPy, and pandas.

## Data preparation

All input files use CSV format. Brightness temperature and LST are expressed in kelvin.

Component-initialization data:

```text
t10,t11,water_vapor,emissivity_10,emissivity_11,a0,a1,a2,a3
```

Simulation or station LST data:

```text
t10,t11,water_vapor,emissivity_10,emissivity_11,lst
```

Station data used for leave-one-site-out validation additionally require:

```text
site_id
```

The ten records in `data/sample.csv` illustrate the required input format.

## Model training

### 1. Component-network initialization

```bash
python train_components.py
```

This stage trains SWBP1-SWBP4 using the component labels `a0`, `a1`, `a2`, and `a3`.

### 2. Simulation-domain joint optimization

```bash
python train_simulation.py
```

This stage loads the four initialized component networks and jointly optimizes them within the complete SW physical backbone using simulated LST.

### 3. Station-observation fine-tuning

```bash
python train_station.py
```

This stage fine-tunes the simulation-trained model using global in situ LST observations.

Data paths, checkpoint paths, training epochs, batch size, and learning rate can be changed through command-line arguments. Run `python <script_name>.py --help` for details.

## Cross-validation

### Simulation-data 10-fold cross-validation

```bash
python validate_simulation.py
```

The simulation samples are divided into ten mutually exclusive folds. For each fold, a new model is trained using the other nine folds and evaluated only on the held-out fold. Every simulation sample is evaluated once.

### Site-based leave-one-site-out cross-validation

```bash
python validate_sites.py
```

In each iteration, all observations from one site are held out. The model is initialized from the same simulation-domain checkpoint, fine-tuned using the remaining sites, and evaluated on the held-out site. The study uses 4,450 Landsat-in situ matchup samples from 29 sites.

The scripts report fold-level and pooled RMSE, MAE, bias, and R² and save the predictions and metrics in `results/`.

## Data availability

The datasets used in this study are publicly available from their original data providers, subject to the corresponding access and citation requirements. The main sources include:

* [USGS Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-1-data)

* [MODIS MOD11A2 LST and emissivity product](https://modis.gsfc.nasa.gov/data/dataprod/mod11.php)

* [ASTER Global Emissivity Dataset](https://lpdaac.usgs.gov/products/ag100v003/)

* [NCEP/NCAR Reanalysis](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html)

* [Baseline Surface Radiation Network (BSRN)](https://bsrn.awi.de/project/the-state-of-affairs/)

* [Surface Radiation Budget Network (SURFRAD)](https://gml.noaa.gov/grad/surfrad/)

The atmospheric-profile data (GAPRI and TIGR) and HIWATER observations can be obtained from the data sources described and cited in the paper. Please refer to Section 2 of the paper for the dataset selection, preprocessing, temporal matching, and quality-control procedures used in this study.

The repository includes only ten example records to demonstrate the input format. Users can reconstruct the research dataset from the public sources by following the procedures described in the paper.

## Evaluation metrics

* **R²**: coefficient of determination.

* **MAE**: mean absolute error.

* **RMSE**: root mean square error.

* **Bias**: mean difference between retrieved and reference LST.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{xie2025mechanismcoupledsplitwindownetwork,
  title={A Mechanism-Coupled Split Window Network for Medium- to High-Resolution Land Surface Temperature Retrieval},
  author={Tian Xie and Menghui Jiang and Chao Zeng and Huifang Li and Guanhao Zhang and Chan Li and Huanfeng Shen},
  year={2025},
  eprint={2509.04991},
  archivePrefix={arXiv},
  primaryClass={physics.ao-ph},
  url={https://arxiv.org/abs/2509.04991}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
