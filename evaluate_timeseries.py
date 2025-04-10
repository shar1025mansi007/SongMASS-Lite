#!/usr/bin/env python
import argparse
import dtw
import numpy as np
from utils import get_songs

parser = argparse.ArgumentParser(description="Time series evaluation for melody.")

parser.add_argument('--lyric-file', type=str, required=True,
                    help="The ground truth lyric file (e.g., lyric.gt)")
parser.add_argument('--melody-file', type=str, required=True,
                    help="The ground truth melody file (e.g., melody.gt)")
parser.add_argument('--song-id-file', type=str, required=True,
                    help="The ground truth song id file (e.g., song_id_test.txt)")
parser.add_argument('--generated-file', type=str, required=True,
                    help="Path to the generated melody file (e.g., melody_inference_results.txt)")

# Build duration vocabulary: keys are strings ('129', '130', ..., '260')
duration_vocab = {str(129 + i): x / 100 for i, x in enumerate(range(25, 3325, 25))}

def flatten(notes, ignore_rest=False):
    """
    Convert a list of (pitch, duration) tuples into a flattened list
    where each pitch is repeated proportional to its duration.
    If a duration token isn't found in the duration vocabulary, a warning is printed and the note is skipped.
    """
    flatten_notes = []
    for note in notes:
        # note is a tuple: (pitch_str, duration_str)
        try:
            pitch = int(note[0])
        except Exception as e:
            print("Error converting pitch in note:", note)
            continue
        try:
            duration_val = duration_vocab[str(note[1])]
        except KeyError:
            print("Warning: duration token not found in duration_vocab:", note[1])
            continue
        if pitch == 128:
            if ignore_rest or not flatten_notes:
                continue
            flatten_notes.extend([flatten_notes[-1]] * int(duration_val * 4))
        else:
            flatten_notes.extend([pitch] * int(duration_val * 4))
    return flatten_notes

def sample_notes(flatten_notes, freq=2):
    return [flatten_notes[i * freq] for i in range(len(flatten_notes) // freq)]

def main():
    args = parser.parse_args()
    
    # Get generated (hypotheses) songs and ground truth songs using utility function.
    hypos = get_songs(
        args.generated_file,
        lyric_file=args.lyric_file,
        song_id_file=args.song_id_file,
        is_generated=True,
        get_last=True,
        find_structure=True,
        cut_exceed_sent=True,
    )
    targets = get_songs(
        args.melody_file,
        lyric_file=args.lyric_file,
        song_id_file=args.song_id_file,
    )
    
    # Flatten each song into a sequence of pitches.
    flatten_hypos = list(map(flatten, hypos))
    flatten_targets = list(map(flatten, targets))
    
    # Sample (downsample) the flattened sequences for DTW computation.
    flatten_hypo_samples = list(map(sample_notes, flatten_hypos))
    flatten_target_samples = list(map(sample_notes, flatten_targets))
    
    dtw_mean = []
    for i in range(len(targets)):
        if len(flatten_target_samples[i]) == 0 or len(flatten_hypo_samples[i]) == 0:
            continue
        d1 = np.array(flatten_target_samples[i]).reshape(-1, 1)
        d2 = np.array(flatten_hypo_samples[i]).reshape(-1, 1)
        d1 = d1 - np.mean(d1)
        d2 = d2 - np.mean(d2)
        d, _, _, _ = dtw.accelerated_dtw(d1, d2, dist='euclidean')
        dtw_mean.append(d / len(d2))
    
    if dtw_mean:
        print('The melody distance is {}.'.format(sum(dtw_mean) / len(dtw_mean)))
    else:
        print("No valid data for DTW evaluation.")

if __name__ == '__main__':
    main()
