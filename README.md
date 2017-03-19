# Relations between ideas

This project provides a framework to identify relations between ideas in temporal text corpora.
(copy some intro content here)

### Main idea
The main idea of this framework is to combine cooccurrence within documents and prevalence correlation over time.

Refer to this writeup for our initial exploration using this framework on news issues and research papers

### Usage

```
python main.py --input_file [data.jsonlist] --data_output_dir [intermediate data directory] \
           --final_output_dir [final output directory] --option [topics or]
```

In the output directory, there will two sub directories, figure/ and table/ and also a main.tex file that can be compiled to generate an example report.

All packages in requirements.txt are necessary. 
Run `pip install -r requirements.txt` to install them.

