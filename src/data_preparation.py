import json
from pathlib import Path
from typing import List, Dict

import ftfy

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_FULL_CHAT_FILE,
    RAW_TRAIN_FILE,
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_FILE,
    PROCESSED_VALIDATION_FILE,
    METADATA_FILE,
    DEFAULT_VALIDATION_SIZE,
    DEFAULT_RANDOM_STATE,
)


SYSTEM_INSTRUCTION = (
    "You are an Alzheimer's support chatbot. "
    "Give helpful, simple, compassionate, and safe information for patients and caregivers. "
    "Do not claim to diagnose disease. "
    "Do not replace doctors. "
    "For emergencies or medication decisions, advise contacting a qualified healthcare professional."
)


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """
    Reads CSV using common encodings.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as error:
            last_error = error

    raise RuntimeError(f"Could not read CSV file: {file_path}. Error: {last_error}")


def normalize_text(value) -> str:
    """
    Cleans text values and repairs broken encoding (mojibake).
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = ftfy.fix_text(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def load_full_chat_dataset(file_path: Path) -> pd.DataFrame:
    """
    Expected columns:
    - Questions
    - Answers
    """
    df = read_csv_safely(file_path)

    required_columns = ["Questions", "Answers"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"{file_path.name} must contain column '{column}'. "
                f"Found columns: {list(df.columns)}"
            )

    new_df = pd.DataFrame()
    new_df["question"] = df["Questions"].apply(normalize_text)
    new_df["answer"] = df["Answers"].apply(normalize_text)
    new_df["source"] = file_path.name

    return new_df


def load_train_dataset(file_path: Path) -> pd.DataFrame:
    """
    Expected columns:
    - Context
    - Response
    """
    df = read_csv_safely(file_path)

    required_columns = ["Context", "Response"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"{file_path.name} must contain column '{column}'. "
                f"Found columns: {list(df.columns)}"
            )

    new_df = pd.DataFrame()
    new_df["question"] = df["Context"].apply(normalize_text)
    new_df["answer"] = df["Response"].apply(normalize_text)
    new_df["source"] = file_path.name

    return new_df


def format_training_text(question: str, answer: str) -> str:
    """
    Prompt format for GPT-style fine-tuning.
    """
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"### User:\n{question}\n\n"
        f"### Assistant:\n{answer}\n"
        f"<|endoftext|>"
    )


def prepare_combined_dataset() -> pd.DataFrame:
    """
    Loads both CSV files, cleans, combines, removes bad rows and duplicates.
    """
    all_frames: List[pd.DataFrame] = []

    if RAW_FULL_CHAT_FILE.exists():
        full_chat_df = load_full_chat_dataset(RAW_FULL_CHAT_FILE)
        all_frames.append(full_chat_df)
    else:
        print(f"Warning: Missing file {RAW_FULL_CHAT_FILE}")

    if RAW_TRAIN_FILE.exists():
        train_df = load_train_dataset(RAW_TRAIN_FILE)
        all_frames.append(train_df)
    else:
        print(f"Warning: Missing file {RAW_TRAIN_FILE}")

    if not all_frames:
        raise RuntimeError("No dataset files found. Please place CSV files in the data folder.")

    combined_df = pd.concat(all_frames, ignore_index=True)

    combined_df["question"] = combined_df["question"].apply(normalize_text)
    combined_df["answer"] = combined_df["answer"].apply(normalize_text)

    combined_df = combined_df[
        (combined_df["question"].str.len() > 2)
        & (combined_df["answer"].str.len() > 2)
    ]

    combined_df = combined_df.drop_duplicates(subset=["question", "answer"])
    combined_df = combined_df.reset_index(drop=True)

    combined_df["text"] = combined_df.apply(
        lambda row: format_training_text(row["question"], row["answer"]),
        axis=1,
    )

    return combined_df


def save_jsonl(records: List[Dict], file_path: Path) -> None:
    """
    Saves list of dictionaries to JSONL.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_and_save_dataset(
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    """
    Main data preparation function.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    combined_df = prepare_combined_dataset()

    train_df, validation_df = train_test_split(
        combined_df,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )

    train_records = train_df[["question", "answer", "source", "text"]].to_dict(orient="records")
    validation_records = validation_df[["question", "answer", "source", "text"]].to_dict(orient="records")

    save_jsonl(train_records, PROCESSED_TRAIN_FILE)
    save_jsonl(validation_records, PROCESSED_VALIDATION_FILE)

    metadata = {
        "total_records": int(len(combined_df)),
        "train_records": int(len(train_df)),
        "validation_records": int(len(validation_df)),
        "columns": ["question", "answer", "source", "text"],
        "train_file": str(PROCESSED_TRAIN_FILE),
        "validation_file": str(PROCESSED_VALIDATION_FILE),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("Dataset preparation completed.")
    print(f"Total records: {metadata['total_records']}")
    print(f"Train records: {metadata['train_records']}")
    print(f"Validation records: {metadata['validation_records']}")
    print(f"Saved train file: {PROCESSED_TRAIN_FILE}")
    print(f"Saved validation file: {PROCESSED_VALIDATION_FILE}")
    print(f"Saved metadata file: {METADATA_FILE}")