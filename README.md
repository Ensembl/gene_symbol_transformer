# GST - Gene Symbol Transformer

A Transformer model for gene symbol assignment of protein coding gene sequences.

The classification model is built for assigning gene symbols and names to protein coding gene sequences from the Ensembl Genebuild annotation.

The network has been trained on a dataset constructed from existing protein coding gene sequences with assigned symbols on the [Ensembl main release](https://www.ensembl.org/), which have been generated either by manual annotation or using an HMM homology method.


## network architecture

![network architecture](images/network_architecture.png?raw=true "Gene Symbol Transformer network architecture")

The training pipeline utilizes a Transformer to learn higher-dimensional features of the raw sequences and performs multiclass classification on the ~30,500 gene symbols that are most frequently occurring in the dataset.


## performance

![assignments vs probability threshold](images/assignments_vs_probability_threshold.png?raw=true "numbe of assignments probability threshold")

The Transformer network exhibits high accuracy in assigning gene symbols. The highest probability gene symbol prediction can be used as a cutoff threshold for the gene symbols to be included in the final assignments, to take into account the network's confidence of the prediction. A probability of value `0.9` is used as the cutoff threshold for assigning gene symbols in production.


## run GST with Docker

Download a pretrained transformer model to assign gene symbols to gene sequences saved in FASTA file format.
```
docker run --read-only \
    --volume="<checkpoints directory path>":/app/checkpoints \
    --volume="<sequences file directory path>":/app/data \
    ensemblorg/gene_symbol_classifier:0.12.1 \
    --checkpoint "/app/checkpoints/<checkpoint filename>" \
    --sequences_fasta "/app/data/<sequences fasta filename>" \
    --scientific_name "<species scientific name>"
```


## License

[Apache License 2.0](LICENSE)
