# Relations between ideas

This project provides a framework to identify relations between ideas in temporal text corpora.
(copy some intro content here)

### Main idea
The main idea of this framework is to combine cooccurrence within documents and prevalence correlation over time.

Refer to this writeup for our initial exploration using this framework on news issues and research papers

### Usage

```
python main.py [--option {topics,keywords}] [--input_file INPUT_FILE] \
               [--data_output_dir DATA_OUTPUT_DIR] \
               [--final_output_dir FINAL_OUTPUT_DIR] \
               [--mallet_bin_dir MALLET_BIN_DIR] \
               [--background_file BACKGROUND_FILE] [--prefix PREFIX] \
               [--num_ideas NUM_IDEAS] \
               [--tokenize] [--lemmatize] [--nostopwords]
```

`input_file` is a data file, where each line corresponds to a document and is a json object with two fields (`text` for the content; `date` for the timestamp, e.g., 20000101).

In the `final_output_dir`, there will two sub directories, figure/ and table/ and also a tex file that is compiled to generate a report pdf file.

If `topics` is used to represent ideas, [Mallet](http://mallet.cs.umass.edu/) is required to generate topics for each document (`mallet_bin_dir` is required).
Alternatively, it works as long as there are mallet style topic files in `data_outpu_dir`.

If `keywords` is used to represent ideas, `background_file` is required to learn `num_ideas` as representations.

`tokenize`, `lemmatize` and `nostopwords` are preprocessing options that we currently support.

All packages in requirements.txt are necessary. 
Run `pip install -r requirements.txt` to install them.

