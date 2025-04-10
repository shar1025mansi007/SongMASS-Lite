#!/usr/bin/env python
import argparse
import numpy as np
from utils import get_songs

parser = argparse.ArgumentParser(description="Histogram evaluation for melody.")
parser.add_argument('--lyric-file', type=str, required=True,
                    help="Path to the ground truth lyric file (e.g., lyric.gt)")
parser.add_argument('--melody-file', type=str, required=True,
                    help="Path to the ground truth melody file (e.g., melody.gt)")
parser.add_argument('--song-id-file', type=str, required=True,
                    help="Path to the ground truth song id file (e.g., song_id_test.txt)")
parser.add_argument('--generated-file', type=str, required=True,
                    help="Path to the generated melody file (e.g., inference_results.txt or melody_inference_results.txt)")
parser.add_argument('--metric', choices=['pitch', 'duration'], default='pitch',
                    help="Calculate the pitch/duration distribution similarity")

def get_pitch_count(x):
    cnt = [0] * 129
    for i in x:
        cnt[int(i[0])] += 1
    return np.array(cnt)

def get_duration_count(x):
    cnt = [0 for _ in range(25, 3325, 25)]
    for i in x:
        cnt[int(i[1]) - 129] += 1
    return np.array(cnt)

def measure_pitch_similarity(targets, hypos):
    song_num = len(hypos)
    similarity = 0

    def get_pitch_histo(x):
        x = get_pitch_count(x)
        total = np.sum(x) if np.sum(x) > 0 else 1
        return x.astype(np.float32) / total

    for i in range(song_num):
        hypo_histo = get_pitch_histo(hypos[i])
        target_histo = get_pitch_histo(targets[i])
        diff = np.abs(hypo_histo - target_histo)
        overlap = (hypo_histo + target_histo - diff) / 2
        similarity += np.sum(overlap)
    return similarity / song_num

def measure_duration_similarity(targets, hypos):
    song_num = len(hypos)
    similarity = 0

    def get_duration_histo(x):
        x = get_duration_count(x)
        total = np.sum(x) if np.sum(x) > 0 else 1
        return x.astype(np.float32) / total

    for i in range(song_num):
        hypo_histo = get_duration_histo(hypos[i])
        target_histo = get_duration_histo(targets[i])
        diff = np.abs(hypo_histo - target_histo)
        overlap = (hypo_histo + target_histo - diff) / 2
        similarity += np.sum(overlap)
    return similarity / song_num

def main():
    args = parser.parse_args()
    
    # Load the generated songs from the generated file.
    # The get_songs function is set up to check if the file is in Fairseq format.
    # If not, it will simply treat every nonempty line as a generated sequence.
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
    
    metric_func = {
        'pitch': measure_pitch_similarity,
        'duration': measure_duration_similarity,
    }
    
    similarity = metric_func[args.metric](targets, hypos)
    print('The {} distribution similarity is {}'.format(args.metric, similarity))

if __name__ == '__main__':
    main()
