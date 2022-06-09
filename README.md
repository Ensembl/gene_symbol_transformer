# GSC - Gene Symbol Classifier

A Transformer model for gene symbol assignment for protein coding gene sequences.

The classification model is built for assigning gene symbols and names to protein coding gene sequences from the Ensembl Genebuild annotation.

The network has been trained on a dataset constructed from existing protein coding gene sequences with assigned symbols on the [Ensembl main release](https://www.ensembl.org/), which have been generated either by manual annotation or using an HMM homology method.


## network architecture

![network architecture](images/network_diagram.png?raw=true "GSC Transformer network diagram")

The training pipeline utilizes a Transformer to learn higher-dimensional features of the raw sequences and performs multiclass classification on the ~30,500 gene symbols that are most frequently occurring in the dataset.


## run GSC with Docker

Download a pretrained classifier model to assign gene symbols to gene sequences saved in FASTA file format.
```
docker run --read-only \
    --volume="<checkpoints directory path>":/app/checkpoints \
    --volume="<sequences file directory path>":/app/data \
    ensemblorg/gene_symbol_classifier:0.12.1 \
    --checkpoint "/app/checkpoints/<checkpoint filename>" \
    --sequences_fasta "/app/data/<sequences fasta filename>" \
    --scientific_name "<species scientific name>"
```


## production usage information

Read more about the background of the project and details on its production usage on Confluence:
https://www.ebi.ac.uk/seqdb/confluence/display/ENSGBD/GSC+-+Gene+Symbol+Classifier


## License

[Apache License 2.0](LICENSE)
