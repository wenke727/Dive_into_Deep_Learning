from pathlib import Path

lst_12 = [
    "hybridize",
    "async-computation",
    "auto-parallelism",
    "hardware",
    "multiple-gpus",
    "multiple-gpus-concise",
    "parameterserver",
]

folder_12 = Path("./12_computational-performance")


lst_13 = [
"image-augmentation",
"fine-tuning",
"bounding-box",
"anchor",
"multiscale-object-detection",
"object-detection-dataset",
"ssd",
"rcnn",
"semantic-segmentation-and-dataset",
"transposed-conv",
"fcn",
"neural-style",
"kaggle-cifar10",
"kaggle-dog",
]

folder_13 = Path("./13_computer-vision")


lst_8 = [
    'sequence',
    'text-preprocessing',
    'language-models-and-dataset',
    'rnn',
    'rnn-scratch',
    'rnn-concise',
    'bptt'
]

folder_8 = Path("./08_recurrent-neural-networks")

lst_9 = [
    'gru',
    'lstm',
    'deep-rnn',
    'bi-rnn',
    'machine-translation-and-dataset',
    'encoder-decoder',
    'seq2seq',
    'beam-search'
]

folder_9 = Path('./09_recurrent-modern')


lst_10 = [
    'attention-cues',
    'nadaraya-waston',
    'attention-scoring-functions',
    'bahdanau-attention',
    'multihead-attention',
    'self-attention-and-positional-encoding',
    'transformer'
]

folder_10 = Path("./10_attention-mechanisms")


def create_python(lst, folder):

    file_header = """#%%
%matplotlib inline
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

"""
    fn_lst = []
    for i, name in enumerate(lst):
        fn = folder / f"{i}_{name}.py"
        fn_lst.append(fn)
        with fn.open('w') as f:
            f.write(file_header)

    return fn_lst


if __name__ == "__main__":
    create_python(lst_10, folder_10)