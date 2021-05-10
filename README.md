# Gene Symbol Classifier

Machine Learning pipeline for gene symbol assignment to protein coding gene sequences.

More information and background for the project:
https://www.ebi.ac.uk/seqdb/confluence/display/ENSGBD/Gene+symbol+classifier


## dataset

The training dataset for the gene symbol classifier consists of the canonical translations of protein coding sequences with an Xref pipeline assignment from the [assemblies](https://www.ensembl.org/info/about/species.html) available in the latest Ensembl release. Assemblies without an assembly accession code where excluded, as they have been annotated many years ago and are consequently of lower quality.

This set contains a little more than 3.2 million canonical translations, and almost 130,000 unique gene symbol labels. Some gene symbols have more than one capitalization variants in the set, so a step merging the variants to the most frequent version was performed, resulting to a little more than 95,000 unique gene symbols. The number of times each symbol occurs in the set is very divergent, from 405 times for the most frequently occurring symbol to just a single occurrence for several symbols. The occurrence frequency distribution has a mean of `33.88`, median `1`, and standard deviation `61.94`.

For classification problems, like the current one with symbol assignment, it is far more challenging for a neural network to learn the pattern of features for labels with only a few training examples versus those with a lot of them. Furthermore, a classification problem with upwards of 200 thousand classes, although tractable, isn't trivial to solve with an acceptable accuracy. Last, there are no standard data augmentation techniques to increase the number of protein sequences as training examples for the less frequently occurring symbols. Consequently, partial training datasets are generated by filtering the sequences by the frequency of their associated symbols. The first six dataset versions contain the ~ 25, 26, 27, 28, 29, and 30 thousand most frequent unique symbols respectively. (More precisely, the cutoff limit is applied to the frequency of the symbols, and the datasets contain a few more unique symbols than the aforementioned rounded numbers.)


## feature engineering

For the current implementation of the classifier, the protein sequences are directly converted to features using an one-hot vector representation. The protein sequences contain all of the 26 IUPAC extended protein letters and the asterisk representing a stop codon. That results to a 27-long one-hot vector for each protein letter in a sequence, and a tensor of length 27 x L for each sequence, where L is the sequence length.


## neural network architecture

The current implementation of the classifier is implemented as a multilayer perceptron, or fully connected feedforward neural network, which contains two layers of a tunable number of connections between them. The ReLU activation function is applied after the first layer and the LogSoftmax (`log(Softmax(x))`) is applied as the final activation function. No dropout was used.

Example printout of a network:
```
FullyConnectedNetwork(
  (input_layer): Linear(in_features=27000, out_features=512, bias=True)
  (output_layer): Linear(in_features=512, out_features=25028, bias=True)
  (relu): ReLU()
  (final_activation): LogSoftmax(dim=1)
)
```

The negative log likelihood loss was applied to the network output as the loss function.

The example protein sequences are of variable length with mean `576.49`, median `442`, and standard deviation `511.25`. In order to generate uniform sized batches all sequences were normalized to a length of `1000`, with either truncating longer sequences or padding shorter ones. Therefore, the final feature tensor being fed to the neural network for each example is of size `27 x 1000`.


## experiment setup

A YAML file has been defined for specifying hyperparameters for an experiment. It has a flat dictionary structure with mostly self-explanatory variables.

```yaml
num_symbols: 25028

sequence_length: 1000

batch_size: 1024

num_connections: 512

dropout_probability: 0

learning_rate: 0.001

num_epochs: 100

random_state: 5
```


## create and use a classifier

### training

Training can be either run directly on a compute node or submitted as an LSF job either manually or using a script that takes just the experiment settings YAML file as an argument.

train directly on a compute node
```
python fully_connected_pipeline.py -ex <experiment settings YAML file path> --train --test

# e.g.
python fully_connected_pipeline.py -ex experiment.yaml --train --test
```

submit a training job with bsub
```
bash submit_training.sh <experiment settings yaml file path>

# e.g.
bash submit_training.sh experiment.yaml
```

### testing

Testing of a trained neural network would normally run right after training. In cases when testing didn't complete, it can be issued separately for the saved training checkpoint file.

load checkpoint and test the trained network
```
python fully_connected_pipeline.py --checkpoint <checkpoint path> --test
```

submit a testing job with bsub
```
bash submit_testing.sh <checkpoint file path>
```

### assign gene symbols to protein sequences

After training, the network is ready to assign gene symbols to protein sequences in a FASTA file which are saved in a CSV file alongside the identifier in the description of each sequence. (For Ensembl protein sequences FASTA files this is the translation stable ID.)

assign symbols to sequences in a FASTA file and save them to a CSV file
```
python fully_connected_pipeline.py --checkpoint <checkpoint path> --sequences_fasta <FASTA file path>
```

### evaluate a trained network

A trained network can be evaluated by assigning gene symbols to the canonical translations of protein sequences of annotations in the latest Ensembl release and comparing them to the existing symbol assignments.

evaluate a trained network
```
python evaluate_network.py --checkpoint <checkpoint path>
```

The gene symbol assignments of a classifier can also be directly compared with the existing gene symbols in an Ensembl release.
```
python evaluate_network.py --assignments_csv <assignments CSV path> --ensembldb_species_database <species database name>
```
