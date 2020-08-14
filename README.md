# Gene Symbol Classifier

https://www.ebi.ac.uk/seqdb/confluence/display/ENSGBD/Gene+symbol+classifier


## 100 most frequent gene symbols prototype

The initial prototype will be a pipeline to select, generate features, and train a neural network with the 100 most frequent gene symbols.

Symbol names sometimes vary in capitalization, and for this reason a lower case version of them have been generated and used for the pipeline.

There are a total of `30907` sequences between the 100 most frequent gene symbols, that have been saved along with their corresponding stable id and symbol in a FASTA file.

A BLAST database have been generated from these sequences and they have subsequently been BLASTed against this database, with the raw BLAST results saved in a [shelve database](https://docs.python.org/3/library/shelve.html) file for easy access, using a FASTA representation of the sequence along with its stable_id and symbol name as the key and the raw results string as the value.
