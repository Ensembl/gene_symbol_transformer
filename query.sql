-- show database tables
/*
SELECT name FROM sqlite_master
WHERE type='table'
ORDER BY name;
/*
*/


/*
SELECT
-- *
COUNT(*)
FROM odb10v1_OGs
LIMIT 10
;
/*
*/


/*
*/
SELECT
-- *
'odb10v1_OGs', odb10v1_OGs.*, 'odb10v1_genes', odb10v1_genes.*, 'odb10v1_gene_xrefs', odb10v1_gene_xrefs.*, 'odb10v1_all_og_fasta', odb10v1_all_og_fasta.*
-- COUNT(*)
FROM odb10v1_OGs
INNER JOIN odb10v1_OG2genes
ON odb10v1_OGs.og_id = odb10v1_OG2genes.og_id
INNER JOIN odb10v1_genes
ON odb10v1_genes.gene_id = odb10v1_OG2genes.gene_id
INNER JOIN odb10v1_gene_xrefs
ON odb10v1_gene_xrefs.gene_id = odb10v1_genes.gene_id
INNER JOIN odb10v1_all_og_fasta
ON odb10v1_all_og_fasta.gene_id = odb10v1_genes.gene_id
-- WHERE odb10v1_gene_xrefs.external_db = 'ENSEMBL'
-- WHERE odb10v1_gene_xrefs.external_id = '?'
LIMIT 10
;
/*
*/


/*
SELECT
-- sequence
COUNT(*)
FROM odb10v1_all_og_fasta
-- WHERE sequence NOT LIKE 'M%'
LIMIT 10
;
/*
*/
