#!/bin/bash


# ibsub -m 32Gb primates_orthologs_one2one.sh


MAMMAL_GIDS=`cp1 ensembl_compara_108 -N -e "SELECT genome_db_id FROM species_set_header JOIN species_set USING(species_set_id) WHERE species_set_id = 83236;"`

mkdir -p mammal_orthologs

for i in $MAMMAL_GIDS; do
    for j in $MAMMAL_GIDS; do
        if (($i < $j)); then
            echo "Dumping orthologs between genome db ids: $i vs. $j"
            cp1 ensembl_compara_108 -e "
            SELECT
                gdb1.name as species1,
                gdb1.assembly as assembly1,
                gdb2.name as species2,
                gdb2.assembly as assembly2,
                gm1.stable_id AS gene1_stable_id,
                sm1.stable_id AS protein1_stable_id,
                gm2.stable_id AS gene2_stable_id,
                sm2.stable_id AS protein2_stable_id,
                h.description AS homology_type,
                s1.sequence AS prot_seq1,
                s2.sequence AS prot_seq2,
                hm1.perc_id AS perc_id1,
                hm2.perc_id AS perc_id2
            FROM homology_member hm1
            JOIN homology_member hm2
                ON hm1.homology_id = hm2.homology_id
            JOIN homology h
                ON hm1.homology_id = h.homology_id
            JOIN gene_member gm1
                ON hm1.gene_member_id = gm1.gene_member_id
            JOIN gene_member gm2
                ON hm2.gene_member_id = gm2.gene_member_id
            JOIN genome_db gdb1
                ON gm1.genome_db_id = gdb1.genome_db_id
            JOIN genome_db gdb2
                ON gm2.genome_db_id = gdb2.genome_db_id
            JOIN seq_member sm1
                ON hm1.seq_member_id = sm1.seq_member_id
            JOIN seq_member sm2
                ON hm2.seq_member_id = sm2.seq_member_id
            JOIN sequence s1
                ON sm1.sequence_id = s1.sequence_id
            JOIN sequence s2
                ON sm2.sequence_id = s2.sequence_id
            WHERE h.description = 'ortholog_one2one'
                AND gm1.biotype_group = 'coding'
                AND gm1.genome_db_id = $i
                AND gm2.genome_db_id = $j
            ;" > mammal_orthologs/orthologs_one2one_${i}_${j}.tsv
        fi
    done
done
