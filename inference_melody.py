#!/usr/bin/env python
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser(
        description="Generate melody inferences from lyric input using a fine-tuned seq2seq model."
    )
    parser.add_argument("--data_dir", type=str, default="processed_dataset",
                        help="Path to the processed dataset directory.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the saved model checkpoint.")
    parser.add_argument("--gen_subset", type=str, default="valid",
                        help="Which subset to run inference on (e.g., 'valid').")
    parser.add_argument("--beam", type=int, default=5,
                        help="Beam size for generation.")
    parser.add_argument("--nbest", type=int, default=5,
                        help="Number of output sequences to generate per input.")
    parser.add_argument("--max_len", type=int, default=500,
                        help="Maximum generation length.")
    parser.add_argument("--sampling", action="store_true",
                        help="Use sampling for generation instead of beam search.")
    args = parser.parse_args()

    # Load the processed dataset from disk
    dataset = load_from_disk(args.data_dir)
    if args.gen_subset not in dataset:
        raise ValueError(f"Subset '{args.gen_subset}' not found. Available splits: {list(dataset.keys())}")
    subset = dataset[args.gen_subset]

    # Assume the source inputs (lyrics) are stored in the "lyric" column
    input_texts = subset["lyric"]

    # Load model and tokenizer from the checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Build generation parameters; note: if sampling is enabled, do_sample=True
    generate_kwargs = {
        "max_length": args.max_len,
        "num_return_sequences": args.nbest,
        "early_stopping": False,  # corresponds to --no-early-stop
    }
    if args.sampling:
        generate_kwargs["do_sample"] = True
    else:
        generate_kwargs["num_beams"] = args.beam

    outputs = []
    for input_text in input_texts:
        # Tokenize the input lyric text
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.max_len
        )
        # Generate output melodies using the model
        generated_ids = model.generate(**inputs, **generate_kwargs)
        # Decode the generated token IDs into human-readable text
        decoded_outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        outputs.append((input_text, decoded_outputs))
        print("Input (lyric):", input_text)
        for i, out in enumerate(decoded_outputs, start=1):
            print(f"Output {i}:", out)
        print("-" * 40)

    # Optionally, write the results to a file for later review
    with open("melody_inference_results.txt", "w", encoding="utf-8") as f:
        for input_text, decoded_outputs in outputs:
            f.write("Input (lyric): " + input_text + "\n")
            for i, out in enumerate(decoded_outputs, start=1):
                f.write(f"Output {i}: " + out + "\n")
            f.write("\n")

if __name__ == "__main__":
    main()
