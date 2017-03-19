#!/bin/bash

MALLET_BIN_DIR="/home/chenhao/code/Mallet/bin"
DATA_OUPUT_DIR=""
FINAL_OUTPUT_DIR=""
INPUT_FILE=""

python main.py --input_file $INPUT_FILE --data_output_dir $DATA_OUPUT_DIR --final_output_dir $FINAL_OUTPUT_DIR --mallet_bin_dir $MALLET_BIN_DIR --topics

