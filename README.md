# Gene Symbol Classifier

https://www.ebi.ac.uk/seqdb/confluence/display/ENSGBD/Gene+symbol+classifier


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
