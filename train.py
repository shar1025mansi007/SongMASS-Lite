#!/usr/bin/env python
import os
import argparse
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

def main():
    parser = argparse.ArgumentParser(
        description="Quick training run for a seq2seq model using Hugging Face Transformers with dataset subsetting."
    )
    parser.add_argument("--model_name_or_path", type=str, default="sshleifer/distilbart-cnn-12-6",
                        help="Pre-trained model identifier from Hugging Face Hub.")
    parser.add_argument("--dataset_path", type=str, default="processed_dataset",
                        help="Path to the processed dataset directory")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save the final model")
    parser.add_argument("--max_epochs", type=int, default=1,
                        help="Number of training epochs (default: 1 for quick run)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device during training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed precision training (if using GPU)")
    parser.add_argument("--subset_size_train", type=int, default=100,
                        help="If > 0, use only the first N examples for training (default: 100)")
    parser.add_argument("--subset_size_valid", type=int, default=20,
                        help="If > 0, use only the first N examples for validation (default: 20)")
    args = parser.parse_args()

    # Load the tokenized dataset from disk (it should have 'train' and 'valid' splits)
    dataset = load_from_disk(args.dataset_path)

    # Reduce dataset size for a quick run
    if args.subset_size_train > 0:
        dataset["train"] = dataset["train"].select(range(min(len(dataset["train"]), args.subset_size_train)))
    if args.subset_size_valid > 0:
        dataset["valid"] = dataset["valid"].select(range(min(len(dataset["valid"]), args.subset_size_valid)))

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    # Note: We're not enabling gradient checkpointing here to avoid extra recomputation overhead

    # Setup training arguments optimized for speed:
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        weight_decay=0.01,
        num_train_epochs=args.max_epochs,
        predict_with_generate=False,  # disable generation during training to save time
        logging_dir='./logs',
        logging_steps=100,  # log more frequently for quick feedback
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        evaluation_strategy='no',  # disable evaluation during training to reduce overhead
        save_strategy='no'         # disable periodic checkpoint saving
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer
    )

    # Train the model and explicitly save the final model
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
