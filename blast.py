#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
BLAST pipeline and wrapper.
"""


# standard library imports
import pathlib
import shelve
import subprocess
import sys

# third party imports
from Bio import SeqIO

# project imports


data_directory = pathlib.Path("data")


def blast_sequence(fasta_sequence, db, evalue="1e-3", word_size="3", outfmt="7"):
    """
    BLAST a FASTA sequence using the specified BLAST database db.

    Documentation of BLAST arguments used:

    -db <String>
      BLAST database name
       * Incompatible with:  subject, subject_loc

    -evalue <Real>
      Expectation value (E) threshold for saving hits
      Default = `10'

    -word_size <Integer, >=2>
      Word size for wordfinder algorithm

    *** Formatting options
    -outfmt <String>
      alignment view options:
        0 = pairwise,
        1 = query-anchored showing identities,
        2 = query-anchored no identities,
        3 = flat query-anchored, show identities,
        4 = flat query-anchored, no identities,
        5 = XML Blast output,
        6 = tabular,
        7 = tabular with comment lines,
        8 = Text ASN.1,
        9 = Binary ASN.1,
       10 = Comma-separated values,
       11 = BLAST archive format (ASN.1)
       12 = JSON Seqalign output
    """
    arguments = [
        "blastp",
        "-db",
        db,
        "-evalue",
        evalue,
        "-word_size",
        word_size,
        "-outfmt",
        outfmt,
    ]
    completed_process = subprocess.run(arguments, input=fasta_sequence, capture_output=True, text=True)
    output = completed_process.stdout

    return output


def generate_blast_features():
    """
    Generate a database with the BLAST results of all sequences.
    """
    db = data_directory / "blast_databases/most_frequent_100/most_frequent_100"

    fasta_path = data_directory / "most_frequent_100.fasta"
    shelve_db_path = data_directory / "blast_results.db"

    with open(fasta_path) as fasta_file, shelve.open(str(shelve_db_path), flag="c") as blast_results:
        for fasta_record in SeqIO.FastaIO.SimpleFastaParser(fasta_file):
            description = fasta_record[0]
            sequence = fasta_record[1]
            fasta_sequence = (f">{description}\n{sequence}\n")

            # evalue = "10"
            # evalue = "1e-1"
            evalue = "1e-3"
            blast_output = blast_sequence(fasta_sequence, db=db, evalue=evalue)

            blast_results[fasta_sequence] = blast_output
            print(description)


def main():
    """
    main function
    """
    generate_blast_features()


if __name__ == "__main__":
    main()
