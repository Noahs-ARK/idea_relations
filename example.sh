#!/bin/bash

MALLET_BIN_DIR="~/mallet/bin"
INPUT_FILE="data/acl/acl2.jsonlist.gz"
BACKGROUND_FILE="data/nips/nips2.jsonlist.gz"
DATA_OUPUT_DIR="acl_example/temp/"
FINAL_OUTPUT_DIR="acl_example/output/"
PREFIX="acl"

# python main.py --input_file $INPUT_FILE --data_output_dir $DATA_OUPUT_DIR --final_output_dir $FINAL_OUTPUT_DIR --mallet_bin_dir $MALLET_BIN_DIR --option keywords --num_ideas 100 --prefix $PREFIX --background_file $BACKGROUND_FILE
python main.py --input_file $INPUT_FILE --data_output_dir $DATA_OUPUT_DIR --final_output_dir $FINAL_OUTPUT_DIR --mallet_bin_dir $MALLET_BIN_DIR --option topics --num_ideas 50 --prefix $PREFIX

