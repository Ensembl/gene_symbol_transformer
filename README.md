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

`data_wrangling.py`
