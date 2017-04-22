#!/bin/bash

MALLET_BIN_DIR="/home/chenhao/code/Mallet/bin"
INPUT_FILE="../data/test/immigration.jsonlist.gz"
BACKGROUND_FILE="../data/test/immigration_other.jsonlist.gz"
DATA_OUPUT_DIR="../data/test_out_0421/"
FINAL_OUTPUT_DIR="example_output_0421/"
PREFIX="immigration_month"

# python main.py --input_file $INPUT_FILE --data_output_dir $DATA_OUPUT_DIR --final_output_dir $FINAL_OUTPUT_DIR --mallet_bin_dir $MALLET_BIN_DIR --option keywords --num_ideas 100 --prefix $PREFIX --background_file $BACKGROUND_FILE
python main.py --input_file $INPUT_FILE --data_output_dir $DATA_OUPUT_DIR --final_output_dir $FINAL_OUTPUT_DIR --mallet_bin_dir $MALLET_BIN_DIR --option topics --num_ideas 50 --prefix $PREFIX --group_by month

