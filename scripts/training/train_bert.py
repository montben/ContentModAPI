"""
BERT fine-tuning script for multi-label content moderation.

This script fine-tunes a pre-trained BERT model for multi-label classification
on the 8-label content moderation schema.

Usage:
    python scripts/training/train_bert.py --config scripts/training/config.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocessing.label_schema import LABEL_SCHEMA, LABEL_NAMES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiLabelTrainer:
    """Custom trainer for multi-label classification."""

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 6):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load tokenizer and model."""
        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )

        logger.info(f"Model loaded with {self.num_labels} labels")

    def prepare_dataset(self, data_path: str) -> DatasetDict:
        """
        Load and prepare the dataset for training.

        Expected format: CSV with 'text' column and 8 binary label columns
        """
        logger.info(f"Loading dataset from {data_path}")

        # Load the merged dataset
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples")

        # Split into train/val/test
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Tokenize function - convert to list first
        def tokenize_function(examples):
            texts = examples['text']
            if not isinstance(texts, list):
                texts = texts.tolist()
            return self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=512
            )

        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Prepare labels (convert to float for multi-label)
        label_columns = list(LABEL_SCHEMA.keys())

        def prepare_labels(examples):
            labels = []
            for i in range(len(examples['text'])):
                label_vector = [float(examples[col][i]) for col in label_columns]
                labels.append(label_vector)
            return {"labels": labels}

        train_dataset = train_dataset.map(prepare_labels, batched=True)
        val_dataset = val_dataset.map(prepare_labels, batched=True)
        test_dataset = test_dataset.map(prepare_labels, batched=True)

        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

    def compute_metrics(self, eval_pred):
        """Compute metrics for multi-label classification."""
        predictions, labels = eval_pred

        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(torch.tensor(predictions))

        # Convert to binary predictions (threshold = 0.5)
        binary_predictions = (predictions > 0.5).float()

        # Calculate F1 scores
        f1_macro = f1_score(labels, binary_predictions, average='macro')
        f1_micro = f1_score(labels, binary_predictions, average='micro')
        f1_weighted = f1_score(labels, binary_predictions, average='weighted')

        # Per-label F1 scores
        per_label_f1 = f1_score(labels, binary_predictions, average=None)
        per_label_metrics = {}
        for i, label_name in enumerate(LABEL_NAMES.values()):
            per_label_metrics[f'f1_{label_name}'] = per_label_f1[i]

        return {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            **per_label_metrics
        }

    def train(self, dataset: DatasetDict, output_dir: str, config: Dict):
        """Train the model."""
        logger.info("Starting training...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.get('epochs', 3),
            per_device_train_batch_size=config.get('batch_size', 16),
            per_device_eval_batch_size=config.get('eval_batch_size', 32),
            warmup_steps=config.get('warmup_steps', 500),
            weight_decay=config.get('weight_decay', 0.01),
            learning_rate=config.get('learning_rate', 2e-5),
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            report_to="wandb" if config.get('use_wandb', False) else None,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")

        return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train BERT for multi-label content moderation")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to merged dataset CSV")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/bert-multilabel",
                       help="Output directory for trained model")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased",
                       help="Pre-trained model name")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="muzzle-content-moderation",
            config=vars(args)
        )

    # Training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'use_wandb': args.use_wandb
    }

    # Initialize trainer
    trainer = MultiLabelTrainer(model_name=args.model_name)
    trainer.load_model()

    # Prepare dataset
    dataset = trainer.prepare_dataset(args.data_path)

    # Train model
    trained_model = trainer.train(dataset, str(output_dir), config)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trained_model.evaluate(dataset['test'])

    # Save results
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Test results saved to {results_path}")
    logger.info(f"Test F1 Macro: {test_results.get('eval_f1_macro', 'N/A')}")

    if args.use_wandb:
        wandb.log(test_results)
        wandb.finish()


if __name__ == "__main__":
    main()
