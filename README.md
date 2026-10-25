Code for the MSc dissertation *Comprehensive integration of genomic
and proteomic datasets to empower machine learning for equitable discovery of drug mechanisms*, submitted for the **Master in Data Science and Engineering**, Faculdade de Engenharia da Universidade do Porto.

**Author:** Ricardo Pinto · **2026**

---

## Overview

Pharmacogenomic knowledge remains concentrated in a relatively small set of well-characterised variants, with evidence disproportionately derived from individuals of European ancestry. This project aimed to prioritise *uncharacterised* missense variants with potential pharmacogenomic relevance and investigate whether high-priority candidates show population differentiation across ancestry groups.

Because the absence of a documented pharmacogenomic association does not imply that a variant is functionally irrelevant, the problem is formulated as **positive–unlabelled (PU) learning** rather than conventional binary classification. Curated variants from ClinPGx serve as labelled positives, while the remaining missense variants identified in the 1000 Genomes Project are treated as unlabelled, rather than assumed to be negative examples.

![Analysis pipeline](https://github.com/rjcpinto/master-thesis-code/blob/main/fig/pipeline_schema_half.png)


## Repository structure

```
.
├── code/
│   ├── db/                                  data warehouse
│   │   ├── schema/
│   │   │   ├── create_dimensions.sql        dimension tables and their loads
│   │   │   └── create_fact.sql              fact_metrics
│   │   ├── queries/
│   │   │   ├── labeled_clinpgx_dataset_query.sql
│   │   │   └── unlabeled_populatinal_dataset_query.sql
│   │   ├── build_database.py                creates a DuckDB file and executes the schema scripts
│   │   ├── query_db.py                      runs a query and exports the result
│   │   └── query_db_query_file.py           same, reading the query from a .sql file
│   └── src/
│       ├── ensembl_api.py                   Ensembl REST queries for variant identifiers and coordinates
│       ├── dbsnp_query.py                   dbSNP lookups
│       ├── protvar_api_requests.py          batch submission of variant identifiers to the ProtVar API
│       ├── protvar_functional_anno_api.py   re-query of the dedicated ProtVar endpoints for variants
│       │                                    returned with incomplete annotation
│       ├── ml_pipeline.py                   splitting, imputation, unsupervised analysis, estimator tournament
│       ├── tuning_hyperparameters.py        randomised search and holdout evaluation
│       └── discovery_positives.py           scoring of the unlabelled pool
├── envs/
│   └── thesis_env.yml                       conda environment
├── connect_googlecolab_locally.sh
└── .gitignore
```

