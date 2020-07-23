#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
BLAST pipeline and wrapper.
"""


# standard library imports
import subprocess
import sys

# third party imports

# project imports


def blast_sequence(query, db, out, evalue="1e-3", word_size="3", outfmt="7"):
    """
    -query <File_In>
      Input file name
      Default = `-'

    -db <String>
      BLAST database name
       * Incompatible with:  subject, subject_loc

    -out <File_Out>
      Output file name
      Default = `-'

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
        "-query",
        query,
        "-db",
        db,
        "-out",
        out,
        "-evalue",
        evalue,
        "-word_size",
        word_size,
        "-outfmt",
        outfmt,
    ]
    completed_process = subprocess.run(arguments, capture_output=True)
    output = completed_process.stdout.decode("utf-8")

    return output


def main():
    """
    main function
    """
    query = "query.fasta"
    db = "data/blast_databases/most_frequent_100/most_frequent_100"
    # out = "results.txt"
    out = "-"
    # evalue = "1e-1"
    # evalue = "1e-3"
    output = blast_sequence(query=query, db=db, out=out)
    print(output)


if __name__ == "__main__":
    main()
