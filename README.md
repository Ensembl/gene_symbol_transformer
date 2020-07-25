# Gene Symbol Classifier

https://www.ebi.ac.uk/seqdb/confluence/display/ENSGBD/Gene+symbol+classifier


## 100 most frequent gene symbols prototype

The initial prototype will be a pipeline to select, generate features, and train a neural network with the 100 most frequent gene symbols.

Symbol names sometimes vary in capitalization, and for this reason a lower case version of them have been generated and used for the pipeline.

There are a total of `30907` sequences between the 100 most frequent gene symbols, that have been saved along with their corresponding stable id and symbol in a FASTA file.

A BLAST database have been generated from these sequences and they have subsequently been BLASTed against this database, with the raw BLAST results saved in a [shelve database](https://docs.python.org/3/library/shelve.html) file for easy access, using a FASTA representation of the sequence along with its stable_id and symbol name as the key and the raw results string as the value.


## scripts

`get_gene_symbols.pl`: retrieve gene symbols from the target database using the Core API

`generate_blast_database.sh`: generate a BLAST database from gene sequences

`blast_sequence.sh`: blast a sequence against a BLAST database


## run JupyterLab on the cluster

scripts in `dev_scripts/`:
```
bsub_jupyter_server.sh
bsub_reverse_proxy.sh
open_bsub_shell.sh
```


## data wrangling

Merge the metadata file `all_species.csv` and the sequences file `all_species.fa` in the file `all_species_metadata_sequences.csv`.

### `data_wrangling()` function in `data_wrangling.py`, results saved in `metadata_sequences.csv`

- simplify a couple of column names
```
"display_xref.display_id" to "symbol"
"display_xref.db_display_name" to "db_display_name"
```

- ignore capitalization of symbol names
```
267536 unique symbols with original names
233824 unique symbols after removing capitalization
12.6% reduction in unique symbols
```
