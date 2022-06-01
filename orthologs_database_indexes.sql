/*
See the NOTICE file distributed with this work for additional information
regarding copyright ownership.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/


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
