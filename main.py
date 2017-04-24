# -*- coding: utf-8 -*-

import os
import sys
import functools
import preprocessing
import mallet_topics as mt
import fighting_lexicon as fl
import idea_relations as il
import argparse
import logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--option", type=str, choices=["topics", "keywords"],
                    help=(
                        "choose using topics or keywords to represent ideas,"
                        " mallet_bin_dir is required if topic is chosen,"
                        " background_file is required if keywords is chosen."
                    ),
                    default="topics")
parser.add_argument("--input_file",
                    help=("input file, each line is a json object "
                          "with fulldate and text"),
                    type=str)
parser.add_argument("--data_output_dir",
                    help=("output directory for intermedia data"),
                    type=str)
parser.add_argument("--final_output_dir",
                    help=("output directory for final results"),
                    type=str)
parser.add_argument("--mallet_bin_dir",
                    help=("directory with Mallet binaries"),
                    type=str)
parser.add_argument("--background_file",
                    help=("background file to learn important keywords"),
                    type=str)
parser.add_argument("--group_by",
                    help=("binning option for timesteps, supports year, quarter, and month"),
                    type=str,
                    default="year")
parser.add_argument("--prefix",
                    help=("name for the exploration"),
                    type=str,
                    default="exp")
parser.add_argument("--num_ideas",
                    help=("number of ideas, i.e., "
                          "number of topics or keywords"),
                    type=int,
                    default=50)
parser.add_argument("--tokenize",
                    help=("whether to tokenize"),
                    action="store_true")
parser.add_argument("--lemmatize",
                    help=("whether to lemmatize"),
                    action="store_true")
parser.add_argument("--nostopwords",
                    help=("whether to filter stopwords"),
                    action="store_true")
args = parser.parse_args()


def main():
    input_file = os.path.abspath(args.input_file)
    data_output_dir = os.path.abspath(args.data_output_dir)
    final_output_dir = os.path.abspath(args.final_output_dir)
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    # Support some standard preprocessing
    if args.tokenize:
        # tokenize input_file to token_file
        logging.info("tokenizing data")
        token_file = "%s/tokens.jsonlist.gz" % data_output_dir
        func = functools.partial(preprocessing.tokenize,
                                 filter_stopwords=args.nostopwords)
        preprocessing.preprocess_input(input_file, token_file)
        input_file = token_file
    if args.lemmatize:
        # lemmatize input_file to lemma_file
        logging.info("lemmatizing data")
        lemma_file = "%s/lemmas.jsonlist.gz" % data_output_dir
        func = functools.partial(preprocessing.lemmatize,
                                 filter_stopwords=args.nostopwords)
        preprocessing.preprocess_input(input_file, lemma_file)
        input_file = lemma_file
    # generate topics or lexicons
    option = args.option
    num_ideas = args.num_ideas
    cooccur_func = functools.partial(il.generate_cooccurrence_from_int_set,
                                     num_ideas=num_ideas)
    prefix = args.prefix
    if option == "topics":
        logging.info("using topics to represent ideas")
        prefix = "%s_topics" % prefix
        # generate mallet topics        
        mt.get_mallet_input_from_words(input_file, data_output_dir)
        if not mt.check_mallet_directory(data_output_dir):
            # run mallet to prepare topics inputs
            # users can also generate mallet-style topic inputs inputs
            logging.info("running mallet to get topics")
            if not os.path.exists(os.path.join(args.mallet_bin_dir, 'mallet')):
                sys.exit("Error: Unable to find mallet at %s" % args.mallet_bin_dir)
            os.system("./mallet.sh %s %s %d" % (args.mallet_bin_dir,
                                                data_output_dir,
                                                num_ideas))
        # load mallet outputs
        articles, vocab, idea_names = mt.load_articles(input_file,
                                                       data_output_dir)
        table_top = 5
    elif option == "keywords":
        logging.info("using keywords to represent ideas")
        prefix = "%s_keywords" % prefix
        # idenfity keyword ideas using fighting lexicon
        lexicon_file = "%s/fighting_lexicon.txt" % data_output_dir
        other_files = [args.background_file]
        fl.get_top_distinguishing(input_file, other_files, data_output_dir,
                                  lexicon_file)
        # load keywords
        articles, word_set, idea_names = fl.load_word_articles(
            input_file,
            lexicon_file,
            data_output_dir,
            vocab_size=num_ideas)
        table_top = 10
    else:
        logging.error("unsupported idea representations")

    # compute strength between pairs and generate outputs
    il.generate_all_outputs(articles, num_ideas, idea_names, prefix,
                            final_output_dir, cooccur_func,
                            table_top=table_top, group_by=args.group_by)


if __name__ == "__main__":
    main()

