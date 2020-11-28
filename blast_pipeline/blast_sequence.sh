#!/usr/bin/env bash


# Copyright 2020 EMBL-European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# -query <File_In>
#   Input file name
#   Default = `-'
query="query.fasta"

# -db <String>
#   BLAST database name
#    * Incompatible with:  subject, subject_loc
#db="data/blast_databases/all_species/all_species"
db="data/blast_databases/most_frequent_101/most_frequent_101"
#db="data/blast_databases/most_frequent_3/most_frequent_3"

# -out <File_Out>
#   Output file name
#   Default = `-'
#out="-"
out="results.csv"

# -evalue <Real>
#   Expectation value (E) threshold for saving hits
#   Default = `10'
evalue="10"
# Ensembl default
#evalue="1e-1"
#evalue="1e-3"

# -word_size <Integer, >=2>
#   Word size for wordfinder algorithm
# Ensembl default
word_size="3"
# NCBI default
#word_size="6"

# *** Formatting options
# -outfmt <String>
#   alignment view options:
#     0 = pairwise,
#     1 = query-anchored showing identities,
#     2 = query-anchored no identities,
#     3 = flat query-anchored, show identities,
#     4 = flat query-anchored, no identities,
#     5 = XML Blast output,
#     6 = tabular,
#     7 = tabular with comment lines,
#     8 = Text ASN.1,
#     9 = Binary ASN.1,
#    10 = Comma-separated values,
#    11 = BLAST archive format (ASN.1)
#    12 = JSON Seqalign output
outfmt="6"
#outfmt="7"

#blastp -query $query -db $db -out $out -evalue $evalue -word_size $word_size -outfmt $outfmt
blastp -query $query -db $db -out $out -outfmt $outfmt
