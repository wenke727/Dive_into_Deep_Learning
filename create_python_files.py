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


def create_python(lst, folder):

    file_header = """#%%
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
    create_python(lst_13, folder_13)