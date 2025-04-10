#!/usr/bin/env python
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser(
        description="Generate lyric inferences from melody using a fine-tuned seq2seq model."
    )
    parser.add_argument("--data_dir", type=str, default="processed_dataset",
                        help="Path to the processed dataset directory.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the saved model checkpoint.")
    parser.add_argument("--gen_subset", type=str, default="valid",
                        help="Dataset subset to run inference on (e.g., 'valid').")
    parser.add_argument("--beam", type=int, default=5,
                        help="Beam size for generation.")
    parser.add_argument("--nbest", type=int, default=5,
                        help="Number of output sequences to generate per input.")
    parser.add_argument("--max_len", type=int, default=500,
                        help="Maximum generation length.")
    parser.add_argument("--sampling", action="store_true",
                        help="Use sampling instead of beam search.")
    args = parser.parse_args()

    # Load the processed dataset from disk.
    dataset = load_from_disk(args.data_dir)
    if args.gen_subset not in dataset:
        raise ValueError(f"Subset '{args.gen_subset}' not found in dataset splits: {list(dataset.keys())}")
    valid_dataset = dataset[args.gen_subset]

    # Assume the source inputs are stored in the "melody" column.
    # (Adjust this if your dataset uses a different column name.)
    input_texts = valid_dataset["melody"]

    # Load model and tokenizer from the given checkpoint path.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Prepare generation parameters.
    generate_kwargs = {
        "max_length": args.max_len,
        "num_return_sequences": args.nbest,
        "early_stopping": False,   # --no-early-stop
    }
    if args.sampling:
        generate_kwargs["do_sample"] = True
        # When sampling, remove beam search parameters.
    else:
        generate_kwargs["num_beams"] = args.beam

    # Run inference on each input example.
    outputs = []
    for idx, input_text in enumerate(input_texts):
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.max_len
        )
        generated_ids = model.generate(**inputs, **generate_kwargs)
        # Decode the generated sequences; skip special tokens (removing BPE markers).
        decoded_outputs = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        outputs.append((input_text, decoded_outputs))
        # Print out the inference for the current example.
        print(f"Input (melody): {input_text}")
        for i, out in enumerate(decoded_outputs, 1):
            print(f"Output {i}: {out}")
        print("-" * 40)

    # Optionally, save all outputs to a file.
    with open("inference_results.txt", "w", encoding="utf-8") as f:
        for input_text, decoded_outputs in outputs:
            f.write("Input (melody): " + input_text + "\n")
            for i, out in enumerate(decoded_outputs, 1):
                f.write(f"Output {i}: {out}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
