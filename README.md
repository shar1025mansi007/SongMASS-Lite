# SongMASS-Lite
This repository contains a reimplementation of SongMASS – an automatic song writing framework for melody and lyric generation – using the Lakh MIDI Dataset (LMD) and HuggingFace’s Transformers library. This work faithfully reproduces the core ideas of SongMASS, including masked sequence-to-sequence (MASS) pre-training and alignment constraints, and presents a modern, reproducible pipeline based on open-source tools.

## Table of Contents

- Overview
- Installation and Requirements
- Data Preparation
- Preprocessing
- Training
-Inference
- Evaluation
- Directory Structure
- Experiments and Results
- References

## Overview
Automatic melody–lyric generation is a challenging task that requires learning to align lyric syllables with musical note events. In this reimplementation of SongMASS, the focus on two central tasks:

- Lyric-to-Melody (L2M) Generation: Generate a melody sequence given a lyric input.
- Melody-to-Lyric (M2L) Generation: Generate lyric sequences that align with a given melody.

This implementation follows the original SongMASS approach (Sheng et al., 2020) by leveraging MASS pre-training along with an attention-based alignment constraint, but this project re-cast the entire pipeline using HuggingFace Transformers (specifically, using a distilled version of BART). This project utilizes the Lakh MIDI Dataset as the primary data source and include custom scripts for data generation, preprocessing, training, inference, and evaluation.

## Installation and Requirements

The project requires **Python 3.x**. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- **PyTorch**: For model training and inference.
- **Transformers**: HuggingFace library for pretrained sequence-to-sequence models.
- **Datasets**: For data preprocessing.
- **NumPy**: For numerical operations.
- **NLTK**: For text tokenization if needed.
- **dtw**: For Dynamic Time Warping evaluation.

## Data Preparation

Data is generated and organized according to the original SongMASS data generation pipeline. In particular, the following bash scripts are used:

- `generate_data.sh`: Copies dictionary files and prepares lyric and melody files.
- `generate_lmd_dataset.py`: A Python script for processing raw LMD MIDI files into aligned lyric–melody pairs following SongMASS's alignment methods.

LMD dataset is obtained from here. Below is the provided script to parse LMD data in the project experiments.

```bash
git clone https://github.com/yy1lab/Lyrics-Conditioned-Neural-Melody-Generation
DATADIR=Lyrics-Conditioned-Neural-Melody-Generation/lmd-full_MIDI_dataset/Sentence_and_Word_Parsing
OUTPUTDIR=data_org

python data/generate_lmd_dataset.py --lmd-data-dir $DATADIR --output-dir $OUTPUTDIR
bash generate_data.sh $OUTPUTDIR
```
Based on the above scripts, data samples will be generated under the data_org directory. 

### The output directories include:

- `data_org/mono`: Contains monolingual lyric and melody files.
- `data_org/para`: Contains paired lyric and melody files along with associated dictionary files and song ID mappings.

Ground truth evaluation files (e.g., `test.lyric`, `test.melody`, `song_id_test.txt`) should be derived from these outputs for evaluation purposes.

## Preprocessing

The preprocessing step tokenizes the raw lyric and melody files and converts them into a HuggingFace Dataset:

```bash
python preprocess_data.py
```
This script reads the lyric and melody files from `data_org/para`, tokenizes them using a BART-based tokenizer (with a maximum length of 128 tokens), and saves the processed dataset in Arrow format under `processed_dataset/`.

## Training

The training script, `train.py`, fine-tunes a distilled BART model on the processed dataset for both L2M and M2L tasks. For example, to train the model on a subset of data, run:

```bash
python train.py --model_name_or_path sshleifer/distilbart-cnn-12-6 \
                --dataset_path processed_dataset \
                --output_dir "/content/drive/MyDrive/AI Reimplementation/results" \
                --max_epochs 3 \
                --per_device_train_batch_size 4 \
                --learning_rate 5e-5 \
                --subset_size_train 300 \
                --subset_size_valid 60 \
                --fp16
```
Key training features:

- Masked Sequence-to-Sequence (MASS) Pre-training Objective: Improves model understanding via masked token prediction.
- Alignment Constraint: Learning token-to-token (lyric-to-melody) alignment to ensure strict correspondence.
- Gradient Accumulation and FP16: Utilized to handle resource constraints.

## Inference
Inference scripts generate outputs from the trained models. We provide two scripts:
- `inference.py`: For lyric-to-melody generation.
- `inference_melody.py`: For melody-to-lyric generation.

Example command for generating melody from lyrics:

```bash
python inference.py --data_dir processed_dataset --model "/content/drive/MyDrive/AI Reimplementation/results/checkpoint-900" --gen_subset valid --beam 5 --nbest 5 --max_len 500 --sampling
```

The outputs are saved to files such as `inference_results.txt` (for lyric-to-melody) and `melody_inference_results.txt` (for melody-to-lyric).

## Evaluation
This project evaluate's the generated outputs using two metrics:

1. Histogram-based Similarity:
Calculates pitch and duration distribution similarity between generated outputs and ground truth.
Example:

```bash
python evaluate_histo.py --lyric-file test.lyric --melody-file test.melody --song-id-file song_id_test.txt --generated-file inference_results.txt --metric pitch
```

2. Time Series (DTW) Evaluation:
Uses Dynamic Time Warping (DTW) to compute a melody distance that captures differences in melody contours.
Example:

```bash
python evaluate_timeseries.py --lyric-file test.lyric --melody-file test.melody --song-id-file song_id_test.txt --generated-file melody_inference_results.txt
```
## Directory Structure
A typical directory layout for this project might be:

```text
├── data_org
│   ├── mono
│   │   ├── train.lyric
│   │   ├── train.melody
│   │   ├── valid.lyric
│   │   ├── valid.melody
│   │   ├── dict.lyric.txt
│   │   └── dict.melody.txt
│   └── para
│       ├── train.lyric
│       ├── train.melody
│       ├── valid.lyric
│       ├── valid.melody
│       ├── test.lyric
│       ├── test.melody
│       ├── dict.lyric.txt
│       ├── dict.melody.txt
│       ├── song_id_valid.txt
│       └── song_id_test.txt
├── processed_dataset  # Saved after running preprocess_data.py
├── results            # Model checkpoints and final models
├── generate_data.sh
├── generate_lmd_dataset.py
├── preprocess_data.py
├── train.py
├── inference.py
├── inference_melody.py
├── evaluate_histo.py
├── evaluate_timeseries.py
├── utils.py
├── requirements.txt
└── README.md
```

## Experimental Results
High-Level Summary
This reimplementation of SongMASS achieves moderate performance on the lyric-to-melody generation task. Objective metrics on the test set indicate:

- Pitch Distribution Similarity: ~0.47
- Duration Distribution Similarity: ~0.60
- Melody DTW Distance: ~14.26

These results show that while the model captures the overall pitch and rhythmic patterns seen in human-composed music, there remains room for improvement in reproducing the fine-grained details of the melody. Qualitative listening tests further suggest that the generated melodies maintain harmonic consistency and proper alignment with input lyrics, though some nuances are lost.

## Detailed Results
The experiments on a held-out test set produced the following metrics for lyric-to-melody generation:

- Pitch Distribution Similarity:
The overlap between the pitch histograms of the generated and ground truth melodies was computed as 0.4724 on a scale from 0 (no overlap) to 1 (perfect match).

- Duration Distribution Similarity:
The overlap between note-duration histograms was 0.6080, indicating that the rhythmic structure of the generated melody was somewhat closer to the reference.

- Melody DTW Distance:
Using Dynamic Time Warping (DTW) on time-normalized pitch sequences, we obtained an average melody distance of 14.2632. A DTW distance of 0 would be a perfect match; therefore, a value of ~14.26 indicates moderate divergence in fine details between generated outputs and ground truth.

These quantitative outcomes, alongside qualitative inspection, show that our reimplementation preserves the overall structural and stylistic elements of music while still exhibiting some differences in precise note sequences, a result expected with limited paired training data and a distilled model architecture.

## References
1. Sheng, Z., Song, K., Tan, X., Ren, Y., Ye, W., Zhang, S., & Qin, T. (2020). SongMASS: Automatic Song Writing with Pre-training and Alignment Constraint. AAAI.
2. Song, K., Tan, X., Ren, Y., Ye, W., Zhang, S., & Qin, T. (2019). MASS: Masked Sequence to Sequence Pre-training for Language Generation. ICML.
3. Yu, B., Lu, P., Wang, R., Hu, W., Tan, X., Qin, T., Ye, W., Zhang, S., & Liu, T.-Y. (2022). Museformer: Transformer with Fine- and Coarse-Grained Attention for Music Generation. NeurIPS.
4. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS.
5. Huang, C. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, C., Dai, A., & Eck, D. (2018). Music Transformer: Generating Music with Long-Term Structure. ICLR Workshop.
6. Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., & Salakhutdinov, R. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. ACL.
7. Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. EMNLP.
8. Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML
