#!/bin/bash
module load lang/Python/3.6.4-foss-2018a
PATH=$PATH:~/.local/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bejana/tfbootstrap/lib64/

cd datamininglab/
python3 CNN_genres.py --json_file all --image_size 32 > models/genre/results.txt