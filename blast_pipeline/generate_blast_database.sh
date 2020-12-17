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


#gene_database="all_species"
#input_file_path="data/$gene_database.fa"


gene_database="most_frequent_101"
#gene_database="most_frequent_3"
input_file_path="data/$gene_database.fasta"


# create the output directory
mkdir --parents --verbose "data/blast_databases/$gene_database"


# -dbtype <String, `nucl', `prot'>
#   Molecule type of target db
dbtype="prot"

# -in <File_In>
#   Input file/database name
#   Default = `-'
in="$input_file_path"

# -out <String>
#   Name of BLAST database to be created
#   Default = input file name provided to -in argumentRequired if multiple
#   file(s)/database(s) are provided as input
out="data/blast_databases/$gene_database/$gene_database"

makeblastdb -dbtype $dbtype -in $in -out $out
