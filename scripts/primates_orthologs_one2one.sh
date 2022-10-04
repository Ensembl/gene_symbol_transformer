#!/bin/bash


# ibsub -m 32Gb primates_orthologs_one2one.sh


for i in 124 150 153 197 199 200 201 203 204 205 206 207 209 210 216 217 219 221 302 361 465 472 474 476
do
    for j in 124 150 153 197 199 200 201 203 204 205 206 207 209 210 216 217 219 221 302 361 465 472 474 476
    do
        if (($i < $j)) then
            cp1 ensembl_compara_107 -e "SELECT gm1.stable_id AS gene1_stable_id, sm1.stable_id AS protein1_stable_id, gm2.stable_id AS gene2_stable_id, sm2.stable_id AS protein2_stable_id, h.description AS homology_type, s1.sequence AS prot_seq1, s2.sequence AS prot_seq2, hm1.perc_id AS perc_id1, hm2.perc_id AS perc_id2
            FROM homology_member hm1
            JOIN homology_member hm2
                ON hm1.homology_id = hm2.homology_id
            JOIN homology h
                ON hm1.homology_id = h.homology_id
            JOIN gene_member gm1
                ON hm1.gene_member_id = gm1.gene_member_id
            JOIN gene_member gm2
                ON hm2.gene_member_id = gm2.gene_member_id
            JOIN seq_member sm1
                ON hm1.seq_member_id = sm1.seq_member_id
            JOIN seq_member sm2
                ON hm2.seq_member_id = sm2.seq_member_id
            JOIN sequence s1
                ON sm1.sequence_id = s1.sequence_id
            JOIN sequence s2
                ON sm2.sequence_id = s2.sequence_id
            WHERE h.description = 'ortholog_one2one'
                AND gm1.genome_db_id = $i
                AND gm2.genome_db_id = $j
            ORDER BY hm1.homology_id; " >> /homes/ivana/primates_orthologs_one2one.txt
        fi
    done
done


# awk ' /^gene1_stable_id/ && FNR > 1 {next} {print $0} ' primates_orthologs_one2one.txt > primates_orthologs_one2one_w_perc_id.txt
