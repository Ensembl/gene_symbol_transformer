#!/bin/bash


#gene_database="homo_sapiens_core_101_38"
#gene_database="mus_musculus_core_101_38"


#input_file_path="data/$gene_database.fasta"


# get the gene symbols from the gene database and save them to a file if missing
# if [[ ! -f "$input_file_path" ]]; then
#     perl get_gene_symbols.pl -dbname "$gene_database" > "$input_file_path"
# fi


gene_database="all_species"
input_file_path="data/$gene_database.fa"


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
out="data/blast_database/$gene_database"

makeblastdb -dbtype $dbtype -in $in -out $out
