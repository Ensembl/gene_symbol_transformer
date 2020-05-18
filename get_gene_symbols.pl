#!/usr/bin/env perl


# imports
use warnings;
use strict;
use feature 'say';

use Bio::EnsEMBL::Translation;
use Bio::EnsEMBL::DBSQL::DBAdaptor;
use Bio::EnsEMBL::Analysis::Tools::Algorithms::ClusterUtils;
use Getopt::Long qw(:config no_ignore_case);
#use Statistics::Descriptive;
#use PointEstimation;


# global variables
my $DEBUG = 0;
my $coord_system = 'toplevel';
#my $dbname = 'caenorhabditis_elegans_core_101_269';
my $dbname = 'homo_sapiens_core_101_38';
#my $dbname = 'mus_musculus_core_101_38';
#my $dbname = 'pan_paniscus_core_101_1';
#my $dbname = 'sus_scrofa_core_101_111';
#my $dbname = 'bos_indicus_hybrid_core_101_1';
#my $dbname = 'eptatretus_burgeri_core_101_32';

my $user = 'ensro';
my $host = 'mysql-ens-vertannot-staging';
my $port = 4573;
my $pass;


# process command line arguments
my $options = GetOptions ("user|dbuser|u=s" => \$user,
                          "host|dbhost|h=s" => \$host,
                          "port|dbport|P=i" => \$port,
                          "dbname|db|D=s"   => \$dbname,
                          "dbpass|pass|p=s" => \$pass,
                          "DEBUG=i" => \$DEBUG);


# create a database adaptor
my $db = new Bio::EnsEMBL::DBSQL::DBAdaptor(
  -port   => $port,
  -user   => $user,
  -host   => $host,
  -dbname => $dbname,
  -pass   => $pass);


# If we need to use non core dbs, the commented out code below and be used
#my $dna_dbname = 'leanne_asterias_rubens_core_101';
#my $dna_user   = 'ensro';
#my $dna_host   = $ENV{GBS2};#'mysql-ens-vertannot-staging';#$ENV{GBS1};
#my $dna_port   = $ENV{GBP2};#4573; #$ENV{GBP1};
#my $dna_pass;

#my $dna_db = new Bio::EnsEMBL::DBSQL::DBAdaptor(
#  -port    => $dna_port,
#  -user    => $dna_user,
#  -host    => $dna_host,
#  -dbname  => $dna_dbname,
#  -pass    => $dna_pass);

#if($dna_db) {
#  $db->dnadb($dna_db);
#}


my $slice_adaptor = $db->get_SliceAdaptor();
my $gene_adaptor = $db->get_GeneAdaptor();
my $meta_adaptor = $db->get_MetaContainer();

# fetch all slices from the "toplevel" coordinate system alias
my $slices = $slice_adaptor->fetch_all($coord_system);
# get the species name as a string
my $production_name = $meta_adaptor->get_production_name;


# DEBUG variable
my $counter = 0;

foreach my $slice (@$slices) {
  if ($DEBUG) {
    # testing: skip chromosomes other than 4
    unless ($slice->seq_region_name =~ /^\d+$/ && $slice->seq_region_name eq '4') {
      next;
    }
  }

  # get all genes that overlap this slice
  my $genes = $slice->get_all_Genes();

  if ($DEBUG) {
    say "Processing: ".$slice->seq_region_name." (".scalar(@$genes)." genes)";
  }

  foreach my $gene (@$genes) {
    # skip non-protein coding genes
    my $biotype = $gene->biotype;
    unless ($biotype eq 'protein_coding') {
      next;
    }

    # skip genes without a display_xref
    my $display_xref = $gene->display_xref();
    unless ($display_xref) {
      next;
    }

    # get the canonical transcript of the gene;
    # exit if a gene without a canonical transcript is encountered
    my $transcript = $gene->canonical_transcript;
    unless ($transcript) {
      die "Didn't find a canonical";
    }

    # get the number of exons of the transcript
    my $cds_exons = $transcript->get_all_CDS();
    my $cds_exon_count = scalar(@$cds_exons);

    # retrieve the protein sequence
    my $protein_seq = $transcript->translate->seq;

    my $cds_length = length($protein_seq) * 3;

    say ">".$display_xref->display_id()."::".$display_xref->db_display_name()."::".$production_name."::".$cds_exon_count."::".$cds_length;
    say $protein_seq;

    if ($DEBUG) {
      $counter = $counter + 1;
      if ($counter == 2) {
        exit;
      }
    }
  }
}
