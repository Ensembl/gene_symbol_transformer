-- See the NOTICE file distributed with this work for additional information
-- regarding copyright ownership.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.


-- reference: https://v101.orthodb.org/download/README.txt


DROP TABLE IF EXISTS odb10v1_OGs;
DROP TABLE IF EXISTS odb10v1_genes;
DROP TABLE IF EXISTS odb10v1_gene_xrefs;
DROP TABLE IF EXISTS odb10v1_OG2genes;
DROP TABLE IF EXISTS odb10v1_all_og_fasta;


-- Ortho DB orthologous groups
CREATE TABLE odb10v1_OGs (
  -- 1. OG unique id (not stable between releases)
  -- max string length: 17
  og_id CHAR(32),
  -- 2. level tax_id on which the group was built
  -- max string length: 9
  tax_id CHAR(16),
  -- 3. OG name (the most common gene name within the group)
  -- max string length: 203
  og_name VARCHAR(256),

  PRIMARY KEY (og_id)
);


-- Ortho DB genes with some info
CREATE TABLE odb10v1_genes (
  -- 1. Ortho DB unique gene id (not stable between releases)
  -- max string length: 16
  gene_id CHAR(16),
  -- 2. Ortho DB individual organism id
  -- max string length: 9
  organism_id CHAR(16),
  -- 3. protein original sequence id, as downloaded along with the sequence
  -- max string length: 40
  sequence_id CHAR(64),
  -- 4. semicolon separated list of synonyms, evaluated by mapping
  -- max string length: 1039
  synonyms VARCHAR(2048),
  -- 5. Uniprot id, evaluated by mapping
  -- max string length: 10
  uniprot_id CHAR(16),
  -- 6. semicolon separated list of ids from Ensembl, evaluated by mapping
  -- max string length: 2222
  ensembl_ids VARCHAR(4096),
  -- 7. NCBI gid or gene name, evaluated by mapping
  -- max string length: 9
  ncbi_gid_or_gene_name CHAR(16),
  -- 8. description, evaluated by mapping
  -- max string length: 383
  description VARCHAR(512),

  PRIMARY KEY (gene_id)
);


-- UniProt, ENSEMBL, NCBI, GO and InterPro ids associated with Ortho DB gene
CREATE TABLE odb10v1_gene_xrefs (
  -- 1. Ortho DB gene id
  -- max string length: 16
  gene_id CHAR(16),
  -- 2. external gene identifier, either mapped or the original sequence id from Genes table
  -- max string length: 43
  external_id CHAR(64),
  -- 3. external DB name, one of {GOterm, InterPro, NCBIproteinGI, UniProt, ENSEMBL, NCBIgid, NCBIgenename}
  -- max string length: 14
  external_db CHAR(16)
);


-- OGs to genes correspondence
CREATE TABLE odb10v1_OG2genes (
  -- max string length: 17
  -- 1. OG unique id
  og_id CHAR(32),
  -- 2. Ortho DB gene id
  -- max string length: 16
  gene_id CHAR(16)
);


-- AA sequence of the longest isoform for all genes participating in OG, fasta formatted
-- headers with orthodb internal gene id as well as a public id
CREATE TABLE odb10v1_all_og_fasta (
  internal_gene_id CHAR(32),
  public_gene_id CHAR(32),
  sequence VARCHAR(102400)
);
