#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import Counter

def get_pitch_duration_sequence(notes):
    seq = []
    i = 0
    while i < len(notes):
        if notes[i] > 128:
            i += 1
        else:
            if i + 1 >= len(notes):
                break
            if notes[i + 1] <= 128:
                i += 1
            else:
                pitch = str(notes[i])
                duration = str(notes[i + 1])
                seq.extend([pitch, duration])
                i += 2
    return seq

def separate_sentences(x, find_structure=False, SEP='[sep]'):
    """
    x: list of tokens (strings)
    If find_structure is True, we attempt to convert the tokens (after filtering out non-digit tokens)
    into integers and then pair them as (pitch, duration). Special tokens like [align] and [sep] are skipped.
    """
    # Create a copy
    z = x.copy()
    # Find indices where SEP appears
    separate_positions = [k for k, v in enumerate(z) if v == SEP]
    separate_positions.insert(0, -1)
    sents = []
    for i in range(len(separate_positions) - 1):
        u, v = separate_positions[i] + 1, separate_positions[i + 1]
        sent = z[u:v]
        if find_structure:
            # Filter out any token that is not purely numeric.
            # Even though "[align]".isdigit() is False, we add an explicit check.
            sent_numeric = [token for token in sent if token.isdigit()]
            try:
                sent_ints = list(map(int, sent_numeric))
            except ValueError as e:
                print("Error converting tokens to int in sentence:", sent_numeric)
                raise e
            sent = get_pitch_duration_sequence(sent_ints)
        sents.append(sent)
    return sents

def get_lyrics(lyric_file):
    with open(lyric_file, 'r') as input_file:
        lines = input_file.readlines()
    # Each line is split into tokens by whitespace.
    lyrics = [line.rstrip('\n').split(' ') for line in lines]
    return lyrics

def get_song_ids(song_id_file):
    with open(song_id_file, 'r') as input_file:
        song_ids = input_file.readlines()
    song_ids = [int(x.rstrip('\n')) for x in song_ids]
    return song_ids

def get_songs(
    melody_file,
    lyric_file=None,
    song_id_file=None,
    is_generated=False,
    get_last=False,
    find_structure=False,
    cut_exceed_sent=False,
    beam=5,
    SEP='[sep]',
    ALIGN='[align]'
):
    lyrics = get_lyrics(lyric_file)
    song_ids = get_song_ids(song_id_file)
    lyric_sents = [lyric.count(SEP) for lyric in lyrics]

    def to_tuple(x):
        # Remove SEP and ALIGN tokens and pair consecutive tokens.
        pitch_duration = [i for i in x if i != SEP and i != ALIGN]
        pd_tuples = [(pitch_duration[2 * i], pitch_duration[2 * i + 1])
                     for i in range(len(pitch_duration) // 2)]
        return pd_tuples

    with open(melody_file, 'r') as input_file:
        melodies = input_file.readlines()
        if is_generated:
            if any(line.startswith("H-") for line in melodies):
                melodies = [line for line in melodies if line.startswith("H-")]
                if len(melodies) == len(lyrics) * beam:
                    melodies.sort(key=lambda x: (int(x.split('\t')[0].split('-')[1]),
                                                 -float(x.split('\t')[1])))
                    melodies = [x for i, x in enumerate(melodies) if i % beam == 0]
                else:
                    melodies.sort(key=lambda x: int(x.split('\t')[0].split('-')[1]))
            elif any("Output 1:" in line for line in melodies):
                new_melodies = []
                for line in melodies:
                    if "Output 1:" in line:
                        cleaned = line.split("Output 1:")[-1].strip()
                        if cleaned:
                            new_melodies.append(cleaned)
                melodies = new_melodies
            else:
                melodies = [line.strip() for line in melodies if line.strip() and line.strip()[0].isdigit()]
        else:
            melodies = [line.rstrip('\n') for line in melodies]

    # Split melody sequences into tokens (assuming space-separated tokens)
    melody_seqs = [melody.split(' ') for melody in melodies]
    # Keep only tokens that are digits, or exactly SEP or ALIGN.
    melody_seqs = [[token for token in seq if token.isdigit() or token in (SEP, ALIGN)]
                   for seq in melody_seqs]

    if get_last:
        for i in range(len(melody_seqs)):
            if not melody_seqs[i] or melody_seqs[i][-1] != SEP:
                melody_seqs[i].append(SEP)

    melody_seq_sents = [separate_sentences(seq, find_structure=find_structure)
                         for seq in melody_seqs]

    song_seqs = []
    for i, seq in enumerate(melody_seq_sents):
        if cut_exceed_sent and i < len(lyric_sents):
            seq = seq[:lyric_sents[i]]
        song_seq = []
        for sent in seq:
            song_seq.extend(sent)
            song_seq.append(SEP)
        song_seqs.append(song_seq)

    song_num = max(song_ids) + 1
    songs = [[] for _ in range(song_num)]
    for k, v in enumerate(song_ids):
        if k < len(song_seqs):
            songs[v].extend(song_seqs[k])
    songs = list(map(to_tuple, songs))
    return songs

if __name__ == '__main__':
    print("This is a utility module for processing song data. Import its functions instead.")
