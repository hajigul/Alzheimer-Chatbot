import argparse
import json
from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BASE_MODEL_NAME,
    MODEL_OUTPUT_DIR,
    PROCESSED_TRAIN_FILE,
    PROCESSED_VALIDATION_FILE,
    DEFAULT_MAX_LENGTH,
)


class JsonlTextDataset(Dataset):
    def __init__(
        self,
        file_path: Path,
        tokenizer,
        max_length: int = 256,
        max_samples: Optional[int] = None,
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        self.texts: List[str] = []

        with open(self.file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)
                text = record.get("text", "").strip()

                if text:
                    self.texts.append(text)

                if max_samples is not None and len(self.texts) >= max_samples:
                    break

        if not self.texts:
            raise ValueError(f"No valid text records found in {self.file_path}")

        print(f"Loaded {len(self.texts)} records from {self.file_path}")

        print("Tokenizing records...")
        self.encodings = self.tokenizer(
            self.texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
        }


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def print_gpu_info():
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        print("GPU not detected by PyTorch. Training will run on CPU.")

    print("=" * 60)


def train_model(
    model_name: str = BASE_MODEL_NAME,
    output_dir: Path = MODEL_OUTPUT_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
    epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    max_train_samples: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
):
    print("train_model() started.")

    print_gpu_info()

    if not PROCESSED_TRAIN_FILE.exists() or not PROCESSED_VALIDATION_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run this first:\n"
            "python 01_prepare_data.py"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.resize_token_embeddings(len(tokenizer))

    print("Preparing train dataset...")
    train_dataset = JsonlTextDataset(
        file_path=PROCESSED_TRAIN_FILE,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_train_samples,
    )

    print("Preparing validation dataset...")
    validation_dataset = JsonlTextDataset(
        file_path=PROCESSED_VALIDATION_FILE,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_validation_samples,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=use_fp16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Training completed successfully.")
    print(f"Model saved at: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Alzheimer's chatbot model.")

    parser.add_argument("--model-name", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)

    return parser.parse_args()


def main():
    print("src.training main() started.")

    args = parse_args()

    print("Arguments received:")
    print(f"  model_name: {args.model_name}")
    print(f"  max_length: {args.max_length}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  max_train_samples: {args.max_train_samples}")
    print(f"  max_validation_samples: {args.max_validation_samples}")

    train_model(
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
    )


if __name__ == "__main__":
    main()