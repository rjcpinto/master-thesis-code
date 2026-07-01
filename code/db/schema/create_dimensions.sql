
-- Population
CREATE TABLE dim_populational (
    User_input VARCHAR,
    CHROM VARCHAR,
    POS INTEGER,
    REF VARCHAR,
    ALT VARCHAR,
    avsnp151 VARCHAR,
    "Func.refGene" VARCHAR,
    "Gene.refGene" VARCHAR,
    Allele VARCHAR,
    Annotation VARCHAR,
    Annotation_Impact VARCHAR,
    Gene_Name VARCHAR,
    Gene_ID VARCHAR,
    Feature_Type VARCHAR,
    Feature_ID VARCHAR,
    Transcript_BioType VARCHAR,
    Rank VARCHAR,
    HGVS_c VARCHAR,
    Ref_AA VARCHAR,
    Alt_AA VARCHAR,
    cDNA_pos_len VARCHAR,
    CDS_pos_len VARCHAR,
    AA_pos_len VARCHAR,
    Distance VARCHAR,
    ERRORS_WARNINGS_INFO VARCHAR,
    "ExonicFunc.refGene" VARCHAR,
    AF_EUR_unrel FLOAT,
    AF_EAS_unrel FLOAT,
    AF_AMR_unrel FLOAT,
    AF_SAS_unrel FLOAT,
    AF_AFR_unrel FLOAT,
    ACMG_GENE VARCHAR,
    ACMG_PATH VARCHAR,
    dbSNPBuildID INTEGER,
    AA_pos INTEGER,
);

-- Insert data into Population table
INSERT INTO dim_populational 
SELECT 
    User_input, CHROM, POS, REF, ALT, avsnp151, "Func.refGene", "Gene.refGene", Allele, Annotation, Annotation_Impact, 
    Gene_Name, Gene_ID, Feature_Type, Feature_ID, Transcript_BioType, Rank, HGVS_c, Ref_AA, Alt_AA, cDNA_pos_len, 
    CDS_pos_len, AA_pos_len, Distance, ERRORS_WARNINGS_INFO, "ExonicFunc.refGene", 
    AF_EUR_unrel, AF_EAS_unrel, AF_AMR_unrel, AF_SAS_unrel, AF_AFR_unrel, ACMG_GENE, ACMG_PATH, dbSNPBuildID, AA_pos
FROM read_csv('/Users/ricardo/Desktop/MECD/2ndYear/Thesis/datasets_final/pop_data_missense_final.csv', nullstr='NA');



-- Variant-Protein impact
CREATE TABLE dim_variant_protein_impact ( 
    User_input VARCHAR,
    Chromosome VARCHAR,
    Coordinate INTEGER,
    ID VARCHAR,
    Reference_allele VARCHAR,
    Alternative_allele VARCHAR,
    Notes VARCHAR,
    Gene VARCHAR,
    Codon_change VARCHAR,
    Strand VARCHAR,
    Canonical_isoform_transcripts TEXT,
    MANE_transcript VARCHAR,
    "Uniprot_canonical_isoform_(non_canonical)" VARCHAR,
    Alternative_isoform_mappings VARCHAR,
    Protein_name VARCHAR,
    Amino_acid_position INTEGER,
    Amino_acid_change VARCHAR,
    Consequences VARCHAR,
    "Residue_function_(evidence)" VARCHAR,
    "Region_function_(evidence)" VARCHAR,
    Protein_existence_evidence VARCHAR,
    Protein_length INTEGER,
    Entry_last_updated DATE,
    Sequence_last_updated DATE,
    Protein_catalytic_activity VARCHAR,
    Protein_complex VARCHAR,
    Protein_sub_cellular_location VARCHAR,
    Protein_family VARCHAR,
    "Gnomad_allele_freq(ac;an;af)" VARCHAR,
    Genomic_location VARCHAR,
    Cytogenetic_band VARCHAR,
    Other_identifiers_for_the_variant VARCHAR,
    Diseases_associated_with_variant VARCHAR,
    Variants_colocated_at_residue_position VARCHAR,
    Position_in_structures VARCHAR,
);

-- Insert data into Variant Protein Impact table
INSERT INTO dim_variant_protein_impact 
SELECT 
    User_input, Chromosome, Coordinate, ID, Reference_allele, Alternative_allele, Notes, 
    Gene, Codon_change, Strand, Canonical_isoform_transcripts, MANE_transcript, 
    "Uniprot_canonical_isoform_(non_canonical)", Alternative_isoform_mappings, 
    Protein_name, Amino_acid_position, Amino_acid_change, Consequences, 
    "Residue_function_(evidence)", "Region_function_(evidence)", Protein_existence_evidence, 
    Protein_length, Entry_last_updated, Sequence_last_updated, Protein_catalytic_activity, 
    Protein_complex, Protein_sub_cellular_location, Protein_family, 
    "Gnomad_allele_freq(ac;an;af)", Genomic_location, Cytogenetic_band, 
    Other_identifiers_for_the_variant, Diseases_associated_with_variant, 
    Variants_colocated_at_residue_position, Position_in_structures
FROM read_csv('/Users/ricardo/Desktop/MECD/2ndYear/Thesis/datasets_final/final_protvar_data.csv', nullstr='NA');



-- Pharmacogenomics
CREATE TABLE dim_pharmacogenomics (
    User_input VARCHAR,
    "Variant Annotation ID" BIGINT,
    "Variant/Haplotypes" VARCHAR,
    Gene VARCHAR,
    "Drug(s)" VARCHAR,
    PMID INTEGER,
    "Phenotype Category" VARCHAR,
    Significance VARCHAR,
    Notes TEXT,
    Sentence TEXT,
    Alleles VARCHAR,
    "Specialty Population" VARCHAR,
    "Metabolizer types" VARCHAR,
    isPlural VARCHAR,
    "Is/Is Not associated" VARCHAR,
    "Direction of effect" VARCHAR,
    "PD/PK terms" VARCHAR,
    "Multiple drugs And/or" VARCHAR,
    "Population types" VARCHAR,
    "Population Phenotypes or diseases" VARCHAR,
    "Multiple phenotypes or diseases And/or" VARCHAR,
    "Comparison Allele(s) or Genotype(s)" VARCHAR,
    "Comparison Metabolizer types" VARCHAR
);

-- Insert data into Pharmacogenomics table
INSERT INTO dim_pharmacogenomics 
SELECT 
    User_input, "Variant Annotation ID", "Variant/Haplotypes", Gene, "Drug(s)", 
    PMID, "Phenotype Category", Significance, Notes, Sentence, Alleles, 
    "Specialty Population", "Metabolizer types", isPlural, "Is/Is Not associated", 
    "Direction of effect", "PD/PK terms", "Multiple drugs And/or", "Population types", 
    "Population Phenotypes or diseases", "Multiple phenotypes or diseases And/or", 
    "Comparison Allele(s) or Genotype(s)", "Comparison Metabolizer types"
FROM read_csv('/Users/ricardo/Desktop/MECD/2ndYear/Thesis/datasets_final/final_clinpgx_processed_dataset.csv', nullstr='NA');
