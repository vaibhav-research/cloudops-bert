# CloudOpsBERT: Domain-Specific Language Models for Cloud Operations

CloudOpsBERT is an open-source project exploring **domain-adapted transformer models** for **cloud operations log analysis** — specifically anomaly detection, reliability monitoring, and cost optimization.

This project fine-tunes lightweight BERT variants (e.g., DistilBERT) on large-scale system log datasets (HDFS, BGL) and provides ready-to-use models for the research and practitioner community.

---

## 🚀 Motivation

Modern cloud platforms generate massive amounts of logs. Detecting anomalies in these logs is crucial for:
- Ensuring **reliability** (catching failures early),
- Improving **cost efficiency** (identifying waste or misconfigurations),
- Supporting **autonomous operations** (AIOps).

Generic LLMs and BERT models are not optimized for this domain. CloudOpsBERT bridges that gap by:
- Training on **real log datasets** (HDFS, BGL),
- Addressing **imbalanced anomaly detection** with class weighting,
- Publishing **open-source checkpoints** for reproducibility.

---

## 📂 Repository Structure

cloudops-bert/
├── src/                  # Training & inference code
│   ├── train.py          # Fine-tuning script
│   ├── predict.py        # Single-line & batch inference
│   └── utils.py, ...     # Data utilities (split, tokenization, etc.)
├── scripts/              # Helper scripts (setup, inference, evaluation)
├── data/processed/       # Normalized HDFS/BGL datasets (instructions only)
├── best_exports/         # Exported best models (for HF upload)
└── README.md


## 🧪 Training (for reproduction)
Example training on HDFS:

```
python src/train.py \
  --dataset hdfs \
  --model_name models/distilbert-base-uncased \
  --max_length 256 \
  --epochs 3 \
  --train_bs 16 \
  --eval_bs 32 \
  --weight_classes \
  --cross_eval_on bgl \
  --output_dir outputs/cloudopsbert
```
Training uses Hugging Face Trainer, class-weighted loss, and checkpoint saving.
We select the best checkpoint by F1-score on the validation set.

## 🔍 Inference (Pretrained)
Predict anomaly probability for a single log line:
```
python src/predict.py \
  --model_dir vaibhav2507/cloudops-bert \
  --subfolder distributed-storage \
  --text "ERROR dfs.DataNode: Lost connection to namenode"
```
Batch inference (file with one log line per row):

```
python src/predict.py \
  --model_dir vaibhav2507/cloudops-bert \
  --subfolder distributed-storage \
  --file samples/sample_logs.txt \
  --threshold 0.5 \
  --jsonl_out predictions.jsonl
```

## 📊 Results
* distributed-storage (in-domain, test set)
  * F1: 0.571
  * Precision: 0.992
  * Recall: 0.401
  * AUROC: 0.730
  * Threshold: 0.50 (tuneable)
- Cross-domain (distributed-storage → hpc)
- Performance degrades significantly due to dataset/domain shift (see paper).
- BGL (training in progress)
- Will be released as cloudops-bert (subfolder bgl) once full training is complete.

## 📦 Models

* vaibhav2507/cloudops-bert (Hugging Face Hub)
  * subfolder="distributed-storage" – HDFS-trained CloudOpsBERT
  * subfolder="hpc" – BGL-trained CloudOpsBERT (coming soon)
* Each export includes:
  * Model weights (pytorch_model.bin)
  * Config with label mappings (normal, anomaly)
  * Tokenizer files

## 🚀 Quickstart (Scripts)
 1) Setup folders
```
bash scripts/setup_dirs.sh
```

 2) (optional) Download a local copy of a submodel from Hugging Face
```
bash scripts/fetch_pretrained.sh                # downloads 'distributed-storage' by default
SUBFOLDER=hpc bash scripts/fetch_pretrained.sh  # downloads 'hpc'
```

 3) Single-line prediction (directly from HF)
```
bash scripts/predict_line.sh "ERROR dfs.DataNode: Lost connection to namenode" distributed-storage
```

 4) Batch prediction (using local model folder)
```
bash scripts/make_sample_logs.sh
bash scripts/predict_file.sh samples/sample_logs.txt distributed-storage models/cloudops-bert-distributed-storage preds/distributed_storage.jsonl
```

## 📚 Related Work

Several prior works have explored using BERT for log anomaly detection:

- Leveraging BERT and Hugging Face Transformers for Log Anomaly Detection
- Tutorial-style blog post demonstrating how to fine-tune BERT on log data with Hugging Face. Useful as an introduction, but not intended as a reproducible research artifact.

LogBERT (HelenGuohx/logbert)
- Academic prototype from ~2019–2020 focusing on modeling log sequences with BERT. Demonstrates feasibility but limited to in-domain experiments and lacks integration with modern Hugging Face tooling.
  
AnomalyBERT (Jhryu30/AnomalyBERT)
- Another exploratory repository showing BERT-based anomaly detection on logs, with dataset-specific preprocessing. Similar limitations in generalization and reproducibility.

## 🔑 How CloudOpsBERT is different
- Domain-specific adaptation: explicitly trained for cloud operations logs (HDFS, BGL) with class-weighted loss.
- Cross-domain evaluation: includes in-domain and cross-domain benchmarks, highlighting generalization challenges.
- Reproducibility & usability: clean repo, scripts, and ready-to-use Hugging Face exports.
- Future directions: introduces MicroLM — compressed micro-language models for efficient edge/cloud hybrid inference.
- In short: previous work showed that “BERT can work for logs.”
- CloudOpsBERT operationalizes this idea into reproducible benchmarks, public models, and deployable tools for both researchers and practitioners.

## 📜 Citation
If you use CloudOpsBERT in your research or tools, please cite:
```
@misc{pandey2025cloudopsbert,
  title={CloudOpsBERT: Domain-Specific Transformer Models for Cloud Operations Anomaly Detection},
  author={Pandey, Vaibhav},
  year={2025},
  howpublished={GitHub, Hugging Face},
  url={https://github.com/vaibhav-research/cloudops-bert}
}
```


## 🙌 Contributing
Contributions welcome!
Please open issues for bugs, ideas, or dataset integrations.
