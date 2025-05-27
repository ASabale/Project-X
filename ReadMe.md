# Real-Time Fraud Detection – Project README

## Overview

This project implements a real-time machine learning pipeline for detecting fraudulent transactions. It is structured for reproducibility, robust model evaluation, and easy integration with modern data infrastructure using **Apache Spark**, **Apache Kafka**, **Apache Iceberg**, and GPU-accelerated ML frameworks (XGBoost, LightGBM).
The project is organized into clear modules for **EDA**, **Environment Setup**, **Modeling**, and **Scripts**.

---

## Table of Contents

1. [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)
2. [Environment Setup](#environment-setup)
3. [Model](#model)
4. [Scripts](#scripts)
5. [References](#references)

---

## EDA (Exploratory Data Analysis)

* **Dataset:** Synthetic, feature-engineered bank account applications from Feedzai's BAF Dataset Suite.
* **Target:** `fraud_bool` (1 = Fraud, 0 = Legitimate).
* **Key Properties:**

    * **Temporal** data with a `month` column for time-based validation.
    * **Severe class imbalance:** \~1.1% fraudulent.
    * **Features:** Numeric (income, credit score, etc.), categorical, binary, temporal, and engineered features.
* **Analysis Highlights:**

    * Checked and handled missing data (special codes like `-1`).
    * Initial feature importance identified credit risk, income, and velocity features as predictive.
    * Chose **Yeo-Johnson** transformation for numeric and **one-hot encoding** for categorical features.
    * Emphasized need for time-based validation to prevent leakage.
* **Findings:**

    * Temporal validation is critical.
    * Handling class imbalance and monitoring fairness are core requirements.
* See `/EDA/README.md` for detailed insights, charts, and links to data sources.

---

## Environment Setup

* **Core Stack:**

    * **Apache Spark** (distributed processing)
    * **Apache Kafka** (real-time streaming)
    * **Apache Iceberg** (data lake/table management)
    * **MinIO S3** (object storage)
    * **PostgreSQL** (catalog for Iceberg)
    * **Kafka UI** (monitoring)
* **Containerization:** All services run via Docker and Docker Compose for portability.
* **Quick Start:**

    1. Clone the repo.
    2. Use the provided Makefile commands to start, stop, or rebuild services (`make start-iceberg-minio`, etc.).
    3. Services are defined in `docker-compose-minio.yml`.
    4. Persistent volumes and a shared network for seamless integration.
* **Usage:**

    * See `/environment/README.md` for detailed commands, service breakdown, and troubleshooting.

---

## Model

* **ML Pipeline:**

    * **Algorithms:** XGBoost & LightGBM with GPU acceleration.
    * **Splits:** Strictly time-aware (Train: months 0–3, Validation: 4, Dev-Test: 5, Hold-out: 6–7).
    * **Preprocessing:** Yeo-Johnson for numeric, one-hot for categorical; all transformers serialized.
    * **Class Imbalance:** SMOTE on training data only.
    * **Evaluation:** ROC/AUC, AUC-PR, F1, confusion matrix on proper splits.
    * **Reproducibility:** All artifacts (models, transformers, metrics) are versioned in `/models/<timestamp>/`.
    * **Hardware:** Designed for GPU training (Colab T4 GPU, Python 3.11+).
* **Validation Results:**

    * High precision but low recall, as expected from severe class imbalance.
    * Threshold tuning and metric plots are available.
* **Next Steps:**

    * Sklearn pipeline wrappers, full-data retraining, systematic artifact logging.
* See `/model/README.md` for the full pipeline, metrics, code, and environment details.

---

## Scripts

* **Kafka Producer:**

    * Sends data to Kafka topic for simulation of real-time streaming.
    * Requires `confluent-kafka-python` (`pip install confluent-kafka`).
    * Run: `python kafka-producer.py`
* **Kafka-Spark Consumer:**

    * Consumes streamed data and processes with Spark (in container).
    * Submit with:

        * `docker exec -it spark-iceberg spark-submit /home/iceberg/scripts/kafka-spark-consumer.py`
        * Or, with more options (see scripts/README.md for command)
    * All scripts should be accessible in `/home/iceberg/scripts/` inside the container.
* **Notebooks and Utilities:**

    * Jupyter notebooks for EDA and prototyping.
    * Helper scripts for preprocessing, evaluation, and artifact management.
* See `/scripts/README.md` for full details on script usage and orchestration.

---

## References

* [BAF Dataset Suite (GitHub)](https://github.com/feedzai/bank-account-fraud)
* [SMOTE Paper](https://arxiv.org/abs/1106.1813)
* [XGBoost Docs](https://xgboost.readthedocs.io/)
* [LightGBM Docs](https://lightgbm.readthedocs.io/)
* [Imbalanced-learn Docs](https://imbalanced-learn.org/)
* See module READMEs for more links.

---

## Repo Structure

```
/
├── EDA/
│   └── README.md
├── environment/
│   └── README.md
├── model/
│   └── README.md
├── scripts/
│   └── README.md
├── ReadMe.md
└── ...
```

---

## Contact

For questions or collaboration, open an issue or contact the project maintainer.

---

**All code, models, and metrics are reproducible from the scripts and notebooks in this repository.**
**Note:** The BAF dataset is for research only. Do not deploy trained models to production.

---
