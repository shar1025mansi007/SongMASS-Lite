#!/usr/bin/env python
import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Each nonempty line corresponds to one sample.
        return [line.strip() for line in f if line.strip()]

def create_dataset(data_dir, split):
    para_dir = os.path.join(data_dir, 'para')
    lyric_file = os.path.join(para_dir, f"{split}.lyric")
    melody_file = os.path.join(para_dir, f"{split}.melody")
    
    lyrics = load_text_file(lyric_file)
    melodies = load_text_file(melody_file)
    
    if len(lyrics) != len(melodies):
        raise ValueError(f"Mismatch in number of lyric and melody samples for {split}: {len(lyrics)} vs {len(melodies)}")
    
    data = {"lyric": lyrics, "melody": melodies}
    return Dataset.from_dict(data)

def main():
    # Adjust this directory to point to your generated data directory (e.g., "data_org")
    data_dir = "data_org"
    splits = ["train", "valid", "test"]
    dataset_dict = {}
    for split in splits:
        dataset_dict[split] = create_dataset(data_dir, split)
    dataset = DatasetDict(dataset_dict)

    # Load the tokenizer from a pre-trained model (here using BART as an example)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    def preprocess_function(examples):
        # Tokenize the lyric as input...
        inputs = examples["lyric"]
        # ...and tokenize the melody as target text.
        targets = examples["melody"]
        
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
        
        # Use the tokenizer’s target processing for the output sequence.
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, padding="max_length", max_length=128)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply tokenization over the entire dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Save the tokenized dataset (it will be saved in Arrow format)
    tokenized_datasets.save_to_disk("processed_dataset")
    print("Preprocessing complete. Tokenized dataset saved to 'processed_dataset'.")

if __name__ == "__main__":
    main()
