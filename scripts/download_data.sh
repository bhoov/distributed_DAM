#!/bin/bash
if [ ! -f $1/letter.arff ]; then
    wget "https://www.openml.org/data/download/6/dataset_6_letter.arff" -c -O $1/letter.arff
else
    echo "File letter.arff already exists."
fi
