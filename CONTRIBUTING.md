# Contributing

Thank you for being part of this project! This document lists the core contributors and guidance for working on the repo.

## Core Contributors

| Contributor | GitHub |
|---|---|
| Alex Kurkar | [@akurkar07](https://github.com/akurkar07) |
| Ben | [@bbeennyy860-cyber](https://github.com/bbeennyy860-cyber) |
| MS-Mesh | [@MS-Mesh](https://github.com/MS-Mesh) |

## Getting Started

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/akurkar07/arm-hackathon.git
   cd arm-hackathon
   pip install -r requirements.txt
   ```
2. Create a feature branch off `main`:
   ```bash
   git checkout -b your-name/feature-description
   ```
3. Make your changes, run tests, then open a pull request into `main`.

## Workflow

- **Branch naming:** `name/short-description` (e.g. `alex/agent-detector`, `ben/onnx-quantization`)
- **Commit style:** short imperative messages (e.g. `Add INT8 quantization benchmark`)
- **PRs:** all changes go through a pull request — at least one other contributor should review before merging
- **Tests:** run `python -m pytest tests/` before pushing; don’t merge if tests are red

## Areas of Work

| Area | Folder | Description |
|---|---|---|
| Ingestion & normalisation | `ingestion/` | Device polling, calibration, tag resolution |
| AI agent | `agent/` | Anomaly detection, airflow graph, controller, logger |
| Actuation | `actuation/` | Command dispatch, hard bound enforcement |
| Storage | `storage/` | InfluxDB interface, time-series helpers |
| Evaluation | `evaluation/` | Scenario runner, metrics, comparison reports |
| Benchmarks | `benchmarks/` | FP32 vs INT8 Arm Performix results |

## Questions

Open a GitHub issue or message the team directly.
