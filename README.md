# Gene Symbol Classifier

Machine Learning pipeline for gene symbol assignment to protein coding gene sequences of an Ensembl Genebuild annotation implemented with PyTorch.

More information and background for the project:
https://www.ebi.ac.uk/seqdb/confluence/display/ENSGBD/Gene+symbol+classifier


## dataset

The training dataset for the gene symbol classifier consists of the canonical translations of protein coding sequences with an Xref pipeline assignment from the [assemblies](https://www.ensembl.org/info/about/species.html) available in the latest Ensembl release. Assemblies without an assembly accession code where excluded, as they have been annotated many years ago and are of lower quality compared to more recent ones.

This set contains a little more than 3.2 million canonical translations, and almost 130,000 unique gene symbol labels. Some gene symbols have more than one capitalization variants in the set, so a step merging the variants to the most frequent version was performed, resulting to a little more than 95,000 unique gene symbols. The number of times each symbol occurs in the set is very divergent, from 405 times for the most frequently occurring symbol to just a single occurrence for several symbols. The occurrence frequency distribution has a mean of `33.88`, median `1`, and standard deviation `61.94`.

For classification problems, like the current one with symbol assignment, it is far more challenging for a neural network to learn the pattern of features for labels with only a few training examples versus those with a lot of them, or in other words, to train successfully with an imbalanced dataset. Furthermore, a classification problem with upwards of 95 thousand classes, although tractable, isn't trivial to tackle with an acceptable accuracy. Last, there are no standard data augmentation techniques to increase the number of protein sequences as training examples for the less frequently occurring symbols. Consequently, a subset of the original dataset is generated by filtering the sequences by the frequency of their associated symbols, with a cutoff limit of at least `11` occurrences, which corresponds to the `30,241` most frequent unique symbols.


## feature engineering

Two types of features are used for training the neural network, the protein letters of the canonical translation sequences and the canonical translation species clade. Both of them are converted to the one-hot encoding representation. The sequences are converted to one-hot encoding by protein letter, each of them can take as value any of the 26 IUPAC extended protein letters or the asterisk, representing a stop codon. The conversion then results to a 27-long one-hot vector for each protein letter in a sequence, and a matrix of size `27 x L` for each sequence, where `L` is the sequence length. This matrix is subsequently flattened to a vector of length `27 * L`. The clade one-hot vector is a simple translation of the clade categorical data type to the one-hot encoding. These two vectors are concatenated together resulting to a single features vector which is ultimately feed to the neural network during training.


## neural network architecture

Currently, the classifier is implemented as a multilayer perceptron, or fully connected feedforward neural network, which contains two layers with a tunable number of connections between them. The ReLU activation function is applied after the first layer, while Softmax is applied as the final activation function. During training a validation set is being tested and early stopping is used for regularization, halting the training session when the validation loss stops decreasing between epochs.

Example printout of a network:
```
FullyConnectedNetwork(
  (input_layer): Linear(in_features=22724, out_features=512, bias=True)
  (relu): ReLU()
  (output_layer): Linear(in_features=512, out_features=30241, bias=True)
  (final_activation): LogSoftmax(dim=1)
)
```

The negative log likelihood loss is applied to the network output as the loss function.

The example protein sequences are of variable length with mean `580.63`, median `444`, and standard deviation `521.15`. In order to generate uniform sized batches all sequences were normalized to a length of `841`, equal to `mean + 0.5 * standard deviation`, with either truncating longer sequences or padding shorter ones. Therefore, the final feature tensor being fed to the neural network for each example is of size `27 x 841`.


## experiment setup

A YAML file has been defined for specifying hyperparameters for an experiment, using a flat dictionary structure:

```yaml
# experiment filenames prefix
filename_prefix: experiment

# PyTorch random number generator initialization seed
random_seed: 5

num_symbols: 30241

# training, validation, test split
test_ratio: 0.1
validation_ratio: 0.1

sequence_length: 841

# number of connections between fully connected network layers
num_connections: 512

# max norm for gradient clipping
clip_max_norm: 5

dropout_probability: 0

batch_size: 1024

learning_rate: 0.001

# number of epochs without validation loss improvement before training stops
patience: 5

# maximum number of training epochs
max_epochs: 100
```


## create and use a classifier

First, set up a Python virtual environment for the project and install its dependencies:

```
pyenv install 3.9.6

pyenv virtualenv 3.9.6 gene_symbol_classifier

poetry install
```

### generate dataset

The dataset generation encompasses downloading canonical translation protein sequences and metadata from the genome assemblies in the latest Ensembl release. It can be recreated with the following command:
```
python dataset_generation.py --generate_dataset
```

Generate development dataset files if also needed:
```
python dataset_generation.py --save_dev_datasets
```

### training

Training can be either run directly on a compute node or submitted as an LSF job either manually or using a script that takes just the experiment settings YAML file as an argument.

train directly on a compute node
```
python gene_symbol_classifier.py -ex <experiment settings YAML file path> --train --test

# e.g.
python gene_symbol_classifier.py -ex experiment.yaml --train --test
```

submit a training job with bsub
```
python submit_LSF_job.py -ex <experiment settings YAML file path>

# e.g.
python submit_LSF_job.py -ex experiment.yaml
```

Resuming training of a network is also supported. Simply load the saved checkpoint and pass the `--train` argument to continue training the network with the same configuration and hyperparameters.
```
python gene_symbol_classifier.py --checkpoint <checkpoint path> --train --test
```

### testing

Testing of a trained neural network would normally run right after training. In cases when testing didn't complete, it can be issued separately for the saved training checkpoint file.

load checkpoint and test the trained network
```
python gene_symbol_classifier.py --checkpoint <checkpoint path> --test
```

submit a testing job with bsub
```
python submit_LSF_job.py --checkpoint <checkpoint path> --test
```

### evaluate a trained network

A trained network can be evaluated by assigning gene symbols to the canonical translations of protein sequences of annotations in the latest Ensembl release and comparing them to the existing symbol assignments.

evaluate a trained network
```
python gene_symbol_classifier.py --evaluate --checkpoint <checkpoint path>
```

The gene symbol assignments of a classifier can also be directly compared with the existing gene symbols in an Ensembl release.
```
python gene_symbol_classifier.py --assignments_csv <assignments CSV path> --ensembl_database <Ensembl core database name>
```

### assign gene symbols to protein sequences

After training, the network is ready to assign gene symbols to protein sequences in a FASTA file which are saved in a CSV file alongside the identifier in the description of each sequence. (For Ensembl protein sequences FASTA files this is the translation stable ID.)

assign symbols to sequences in a FASTA file and save them to a CSV file
```
python gene_symbol_classifier.py --checkpoint <checkpoint path> --sequences_fasta <FASTA file path> --scientific_name <species scientific name>
```


## production Docker / Singularity image

The recommended way to use a gene symbol classifier in production is packaged as a Docker image and converted to a Singularity image after that.

build a Docker image
```
docker image build --tag williamebi/gene_symbol_classifier:<gene symbol classifier version> --file Dockerfile .
```

As the Docker container needs access to the input checkpoint and FASTA sequences files, the directories of these files along with their filenames are required as arguments in order to set up Docker volumes to these directories.

assign gene symbols with a Docker container
```
CHECKPOINTS_DIRECTORY=<checkpoints directory path>; CHECKPOINT=<checkpoint filename>; SEQUENCES_DIRECTORY=<sequences file directory path>; SEQUENCES=<sequences fasta filename>; SCIENTIFIC_NAME=<species scientific name>; docker run --read-only --volume="$CHECKPOINTS_DIRECTORY":/app/checkpoints --volume="$SEQUENCES_DIRECTORY":/app/data williamebi/gene_symbol_classifier:<gene symbol classifier version> --checkpoint "/app/checkpoints/${CHECKPOINT}" --sequences_fasta "/app/data/${SEQUENCES}" --scientific_name "$SCIENTIFIC_NAME"
```

upload the Docker image to Docker Hub
```
docker push williamebi/gene_symbol_classifier:<gene symbol classifier version>
```

generate a Singularity image from a Docker image at Docker Hub
```
singularity pull docker://williamebi/gene_symbol_classifier:<gene symbol classifier version>
```

assign gene symbols with a Singularity image
```
SINGULARITY_PATH=<Singularity image path>; CHECKPOINTS_DIRECTORY=<checkpoints directory path>; CHECKPOINT=<checkpoint filename>; SEQUENCES_DIRECTORY=<sequences file directory path>; SEQUENCES=<sequences fasta filename>; SCIENTIFIC_NAME=<species scientific name>; singularity run --bind "$CHECKPOINTS_DIRECTORY":/app/checkpoints --bind "$SEQUENCES_DIRECTORY":/app/data "$SINGULARITY_PATH" --checkpoint "/app/checkpoints/${CHECKPOINT}" --sequences_fasta "/app/data/${SEQUENCES}" --scientific_name "$SCIENTIFIC_NAME"
```


## License

[Apache License 2.0](LICENSE)
