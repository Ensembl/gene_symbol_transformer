-- generate indexes

DROP INDEX IF EXISTS og_id_odb10v1_OGs_index;
DROP INDEX IF EXISTS gene_id_odb10v1_genes_index;
DROP INDEX IF EXISTS og_id_odb10v1_OG2genes_index;
DROP INDEX IF EXISTS gene_id_odb10v1_OG2genes_index;
DROP INDEX IF EXISTS gene_id_odb10v1_gene_xrefs_index;
DROP INDEX IF EXISTS gene_id_odb10v1_all_og_fasta_index;


-- odb10v1_OGs
CREATE INDEX og_id_odb10v1_OGs_index ON odb10v1_OGs(og_id);

-- odb10v1_genes
CREATE INDEX gene_id_odb10v1_genes_index ON odb10v1_genes(gene_id);

-- odb10v1_OG2genes
CREATE INDEX og_id_odb10v1_OG2genes_index ON odb10v1_OG2genes(og_id);
CREATE INDEX gene_id_odb10v1_OG2genes_index ON odb10v1_OG2genes(gene_id);

-- odb10v1_gene_xrefs
CREATE INDEX gene_id_odb10v1_gene_xrefs_index ON odb10v1_gene_xrefs(gene_id);

-- odb10v1_all_og_fasta
CREATE INDEX gene_id_odb10v1_all_og_fasta_index ON odb10v1_all_og_fasta(gene_id);


-- -- speeds computation 1000x in some cases
-- logger.info("Running ANALYZE features")
-- ANALYZE features;
