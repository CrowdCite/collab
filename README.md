# Introduction

Setup is done by using Conda (miniconda - no default packages are included). Then using Conda we bootstrap the other dependencies, including node. Then using `npm`, included with the node dependency, we can build the latex extractor.

# Setting up

    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    $ conda env create -f dev.yml

# Activate the environment

    $ conda activate dev

# Building

    $ npm install
    $ npm run build

# Running

    $ node build/latexExtract.js -h

# Merging pdf and src latex directories

    $ node build/mergePL.js <dir1> <dir2> ...

Output is in merged/timestamp/\*.

# Run `is_next.py`

    $ python is_next.py --in-dir inputs/ --out-dir outputs/ --lm-card gpt2 --batch-size 16 --gpus $(seq 0 0) --num-readers 1 --num-writers 1 --win-size 5