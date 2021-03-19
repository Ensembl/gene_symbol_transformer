/*
-- get the number of protein coding genes
SELECT
  COUNT(gene.stable_id)
FROM gene
WHERE biotype = 'protein_coding';
-- 22792
/*
*/


/*
SELECT
  gene.stable_id,
  xref.display_label
FROM gene
INNER JOIN xref
  ON gene.display_xref_id = xref.xref_id
WHERE gene.biotype = 'protein_coding';
/*
*/


/*
SELECT
  transcript.stable_id AS transcript_stable_id,
  xref.display_label AS Xref_symbol
FROM gene
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN xref
  ON gene.display_xref_id = xref.xref_id
WHERE gene.biotype = 'protein_coding';
/*
*/


/*
*/
SELECT
  translation.stable_id AS translation_stable_id,
  xref.display_label AS Xref_symbol
FROM gene
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN translation
  ON transcript.canonical_translation_id = translation.translation_id
INNER JOIN xref
  ON gene.display_xref_id = xref.xref_id
WHERE gene.biotype = 'protein_coding';
/*
*/
