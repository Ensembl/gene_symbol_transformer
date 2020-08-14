# Gene Symbol Classifier

https://www.ebi.ac.uk/seqdb/confluence/display/ENSGBD/Gene+symbol+classifier


## 101 most frequent gene symbols prototype

The initial prototype will be a pipeline to select, generate features, and train a neural network with the 101 most frequent gene symbols, being the gene symbols that label at least 297 genes.

Symbol names sometimes vary in capitalization, and for this reason a lower case version of them have been generated and used for the pipeline.

There are a total of `31204` genes between the 101 most frequent gene symbols that have been saved along with their corresponding stable id and symbol in a FASTA file.

A BLAST database have been generated from these sequences and the sequences have subsequently been BLASTed against this database.


## code structure

`dataset_generation.py` : merge original data files, normalize, cleanup, and filter examples to a single pandas dataframe saved as a pickle file

`feature_generation.py` : generate BLAST and raw sequences features

`blast.py` : function and pipeline to BLAST specified sequences and save the raw results to a shelve database file
