import os

import numpy as np
import wfdb
from tqdm import tqdm

from utils import resample


def parse_rhythm_intervals(ann):
    rhythm_indices = [i for i, aux in enumerate(ann.aux_note) if len(aux.strip()) > 0]

    intervals = []
    for i in range(len(rhythm_indices)):
        start_idx = ann.sample[rhythm_indices[i]]
        label = ann.aux_note[rhythm_indices[i]]

        if i < len(rhythm_indices) - 1:
            end_idx = ann.sample[rhythm_indices[i + 1]] - 1
        else:
            end_idx = ann.sample[-1] if len(ann.sample) > 0 else start_idx

        intervals.append((start_idx, end_idx, label.strip()))
    return intervals


def segment_data_using_intervals(signal, intervals, fs, window_size, af_label):
    length = len(signal)
    win_samples = int(fs * window_size)
    segments = []
    labels = []

    start = 0

    while start + win_samples <= length:
        seg = signal[start:start + win_samples]
        # seg = resample(seg, sample_rate=fs, resample_rate=400)
        segment_end = start + win_samples - 1
        is_af = 0
        for (int_start, int_end, int_label) in intervals:
            if int_end >= start and int_start <= segment_end:
                if af_label in int_label:
                    is_af = 1
                    break

        segments.append(seg)
        labels.append(is_af)
        start += win_samples

    return np.array(segments), np.array(labels)


def load_data(records, normalize, window_size, af_label):
    segments, labels = [], []
    for rec in tqdm(records):
        record = wfdb.rdsamp(os.path.join(data_path, rec))[0]
        sig = record
        ann = wfdb.rdann(os.path.join(data_path, rec), 'atr')
        fs = ann.fs
        intervals = parse_rhythm_intervals(ann)
        channel_data = sig[:, 0]
        segs, labs = segment_data_using_intervals(
            signal=channel_data,
            intervals=intervals,
            fs=fs,
            window_size=window_size,
            af_label=af_label
        )
        segments.append(segs)
        labels.append(labs)

    segments = np.concatenate(segments, axis=0)
    labels = np.concatenate(labels, axis=0)

    if normalize:
        mean = np.mean(segments)
        std = np.std(segments) + 1e-8
        segments = (segments - mean) / std

    return segments, labels


if __name__ == '__main__':
    data_path = './SHDB-AF'

    records = os.listdir(data_path)
    records = [os.path.splitext(record)[0] for record in records if record.endswith('.atr')]

    normalize = True
    fs = 200
    window_size = 10
    af_label = 'AFIB'

    npy_path = './npy_files'
    if not os.path.exists(npy_path):
        os.makedirs(npy_path, exist_ok=True)

    segments, labels = load_data(records, normalize, window_size, af_label)
    np.save(os.path.join(npy_path, 'segments.npy'), segments, allow_pickle=True)
    np.save(os.path.join(npy_path, 'labels.npy'), labels, allow_pickle=True)
