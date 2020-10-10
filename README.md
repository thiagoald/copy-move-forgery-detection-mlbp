# Copy-Move Forgery Detection Using Multiresolution Local Binary Patterns

Attempt to replicate "Copy-move forgery detection using multiresolution local binary patterns" by Reza Davarzani, Khashayar Yaghmaie, Saeed Mozaffari, and Meysam Tapak.

This project was done as an assignment for an [Image Processing class](https://sites.google.com/a/cin.ufpe.br/in1024/) at [CIn-UFPE](cin.ufpe.br).

## Install

### Dependencies

Installed using pip:

- opencv-python
- scikit-image
- scipy
- tqdm

```sh
$ pip install -r requirements.txt
```

## Setup

To replicate the results, you can download the same dataset used by [this paper](http://www.diid.unipa.it/cvip/pdf/ArdizzoneBrunoMazzola.pdf). The files are available [here](http://www.diid.unipa.it/cvip/?page_id=48#CMFD) for download and should be extracted to the root directory of the repository. The tests were made using only the `Dataset 0` folder (the other folders may be deleted).

## Run

Run detection on images in `dataset` and output results to `out`:

```sh
$ mkdir out
$ python main.py dataset/Dataset\ 0/ out
```
