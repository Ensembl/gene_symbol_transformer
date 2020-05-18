#!/bin/bash


# -query <File_In>
#   Input file name
#   Default = `-'
query="data/query.fasta"

# -db <String>
#   BLAST database name
#    * Incompatible with:  subject, subject_loc
db="data/blast_databases/homo_sapiens_core_101_38"

# -out <File_Out>
#   Output file name
#   Default = `-'
out=""
#out="results.txt"

# -evalue <Real>
#   Expectation value (E) threshold for saving hits
#   Default = `10'
#evalue="1e-1"
evalue="1e-3"

# -word_size <Integer, >=2>
#   Word size for wordfinder algorithm
word_size="3"

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
outfmt="0"

blastp -db $db -query $query -evalue $evalue -word_size $word_size -outfmt $outfmt
