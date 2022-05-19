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
