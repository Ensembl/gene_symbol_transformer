#!/bin/bash


#gene_database="all_species"
#input_file_path="data/$gene_database.fa"


#gene_database="most_frequent_101"
gene_database="most_frequent_3"
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
