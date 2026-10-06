"""
RNA editing predictor (BiLSTM + additive attention).

Usage:
    python predict.py -s ACGT...                 # one or more sequences
    python predict.py -f input.fa -o out.tsv     # FASTA file

Input sequences should be 101 nt, with the site of interest at the center
(position 51). Longer sequences are center-cropped; shorter ones are padded
with N (which reduces accuracy).
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow logs

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Flatten, Activation, RepeatVector,
                                     Permute, Multiply, Lambda)
from tensorflow.keras.models import load_model

SEQ_LEN = 101
CLS_THRESHOLD = 0.98
REG_VERSIONS = ["c3", "c5", "c6", "c10"]  # models/reg_<version>.keras
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class AdditiveAttention(layers.Layer):
    def __init__(self, num_dim_pernucleoside: int, **kwargs):
        super().__init__(**kwargs)
        self.num_dim_pernucleoside = num_dim_pernucleoside

    def build(self, input_shape):
        self.W = self.add_weight(name="W", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="b", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = Flatten()(e)
        a = Activation("softmax")(e)
        temp = RepeatVector(self.num_dim_pernucleoside)(a)
        temp = Permute([2, 1])(temp)
        output = Multiply()([x, temp])
        return Lambda(lambda v: K.sum(v, axis=1))(output)

    def get_config(self):
        config = super().get_config()
        config.update({"num_dim_pernucleoside": self.num_dim_pernucleoside})
        return config


def encode_sequence_2d(seq, expected_length=SEQ_LEN):
    """One-hot encode to (expected_length, 4); center-crop or N-pad as needed."""
    mapping = {
        'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'U': [0, 0, 0, 1]
    }
    if len(seq) > expected_length:
        start = (len(seq) - expected_length) // 2
        seq = seq[start: start + expected_length]
    else:
        seq = seq[:expected_length].ljust(expected_length, 'N')
    return np.array([mapping.get(c.upper(), [0, 0, 0, 0]) for c in seq])


def load_models(reg_version):
    custom = {"AdditiveAttention": AdditiveAttention}
    cls_model = load_model(os.path.join(MODEL_DIR, "cls_model.keras"),
                           custom_objects=custom, compile=False)
    reg_model = load_model(os.path.join(MODEL_DIR, f"reg_{reg_version}.keras"),
                           custom_objects=custom, compile=False)
    return cls_model, reg_model


def predict(seqs, cls_model, reg_model):
    """Return (edited_call, probability, editing_level) arrays for a list of sequences."""
    X = np.stack([encode_sequence_2d(s) for s in seqs])
    probs = cls_model.predict(X, verbose=0).ravel()
    calls = (probs > CLS_THRESHOLD).astype(int)
    # regression model was trained on sqrt(editing level)
    levels = np.clip(reg_model.predict(X, verbose=0).ravel(), 0, None) ** 2
    return calls, probs, levels


def read_fasta(path):
    records, name, chunks = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def check_sequence(name, seq):
    bad = set(seq.upper()) - set("ACGTU")
    if len(seq) != SEQ_LEN:
        print(f"Warning: {name} is {len(seq)} nt (expected {SEQ_LEN}); "
              f"{'center-cropped' if len(seq) > SEQ_LEN else 'padded with N'}.",
              file=sys.stderr)
    if bad:
        print(f"Warning: {name} contains non-ACGTU characters {sorted(bad)}; "
              f"encoded as all-zero.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Predict RNA editing from 101-nt sequences.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--seq", nargs="+", help="one or more sequences")
    group.add_argument("-f", "--fasta", help="FASTA file of sequences")
    parser.add_argument("-o", "--out", help="output TSV (default: print to screen)")
    parser.add_argument("-m", "--model", choices=REG_VERSIONS, default="c3",
                        help="regression model c3, c5, c6, c10 (default: c3)")
    args = parser.parse_args()

    if args.fasta:
        records = read_fasta(args.fasta)
    else:
        records = [(f"seq{i + 1}", s) for i, s in enumerate(args.seq)]
    if not records:
        sys.exit("No sequences found.")
    for name, seq in records:
        check_sequence(name, seq)

    cls_model, reg_model = load_models(args.model)
    calls, probs, levels = predict([s for _, s in records], cls_model, reg_model)

    out = open(args.out, "w") if args.out else sys.stdout
    out.write("id\tEdited_Prediction\tConfidence_score\tEditing_level\n")
    for (name, _), c, p, l in zip(records, calls, probs, levels):
        out.write(f"{name}\t{c}\t{p:.4f}\t{l:.4f}\n")
    if args.out:
        out.close()
        print(f"Wrote {len(records)} predictions to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
