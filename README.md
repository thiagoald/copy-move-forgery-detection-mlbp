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
- ransac

```sh
$ pip install -r requirements.txt
```

## Setup

Any dataset containing images produced by copy-move forgery should do. To replicate my results, you can download the same dataset used by [this paper](http://www.diid.unipa.it/cvip/pdf/ArdizzoneBrunoMazzola.pdf).

## Run

Run detection on images in `dataset` and output results to `out`:

```sh
$ mkdir out
$ python main.py dataset out
```