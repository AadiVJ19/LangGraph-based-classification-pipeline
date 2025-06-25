# LangGraph-based-classification-pipeline
## Overview
This project implements a robust text classification pipeline using a fine-tuned transformer model and a LangGraph DAG with a self-healing fallback mechanism. It supports human-in-the-loop workflows for high reliability.

## Features
- Fine-tuned DistilBERT for text classification (sentiment analysis by default)
- LangGraph DAG with Inference, Confidence Check, and Fallback nodes
- CLI interface for user input, clarifications, and outputs
- Structured logging of predictions, fallbacks, and final decisions
- (Bonus) Backup model and statistics tracking

## Setup
```bash
pip install -r requirements.txt
pip install transformers peft datasets torch langgraph rich scikit-learn huggingface_hub loguru
```

## Fine-tuning the Model
```bash
python src/fine_tune.py --dataset imdb --output_dir models/distilbert-imdb
```

## Running the Pipeline
```bash
python src/cli.py --model_path models/distilbert-imdb
```

## Files
- `src/fine_tune.py`: Fine-tune the transformer model
- `src/dag.py`: LangGraph DAG definition
- `src/cli.py`: CLI interface
- `src/nodes.py`: Node implementations
- `src/utils.py`: Utilities (logging, etc.)
