# protein-search
Semantic similarity search for proteins.

## Installation

To install the package for a GPU system, run the following command:
```bash
git clone git@github.com:braceal/protein-search.git
cd protein-search
pip install -e .
pip install faiss-gpu==1.7.2
```

## Usage

### CATH Example
We provide an example of how to use the package to search the CATH database.
For convenience, we provide a pre-built index and embeddings for the CATH database.
To use the pre-built index and embeddings, please skip to the `protein-search search-index` command.


First download the CATH database, run the following command:
```bash
protein-search download-dataset --dataset cath --output_dir data/cath
```

Then to create embeddings, run the following command:
```bash
nohup python -m protein_search.distributed_inference --config examples/cath/cath_esm_8m_polaris.yaml &> nohup.out &
```

To build the search index, run the following command:
```bash
protein-search build-index --fasta_dir data/cath/ --embedding_dir examples/cath/cath_esm_8m_embeddings/embeddings --dataset_dir examples/cath/cath_esm_8m_faiss
```

To search the index, run the following command:
```bash
protein-search search-index --dataset_dir examples/cath/cath_esm_8m_faiss --query_file examples/cath/faiss-test-cath-20.fasta --top_k 1
```

Which should output the following:
```console
scores: [0.01191352], indices: [0], tags: ['cath|4_2_0|12asA00/4-330']
scores: [0.03016754], indices: [1], tags: ['cath|4_2_0|132lA00/2-129']
```

### Swiss-Prot Example
This example demonstrates how to use the package to search the Swiss-Prot database.

A copy of the dataset is located on Polaris@ALCF here:
```bash
/lus/eagle/projects/FoundEpidem/braceal/projects/kbase-protein-search/data/sprot
```

Then to create embeddings, run the following command:
```bash
nohup python -m protein_search.distributed_inference --config examples/sprot/sprot_esm_8m.yaml &> nohup.out &
```

To build the search index, run the following command:
```bash
protein-search build-index --fasta_dir /lus/eagle/projects/FoundEpidem/braceal/projects/kbase-protein-search/data/sprot --embedding_dir examples/sprot/sprot_esm_8m_embeddings/embeddings --dataset_dir examples/sprot/sprot_esm_8m_faiss
```

To search the index, run the following command:
```bash
protein-search search-index --dataset_dir examples/sprot/sprot_esm_8m_faiss --query_file examples/sprot/faiss-test-sprot.fasta --top_k 1
```

Which should output the following:
```console
scores: [0.00508352], indices: [467483], tags: ['Q5HAN0']
scores: [0.01382443], indices: [467484], tags: ['Q5AYI7']
```

### Converting Uniprot XML to FASTA
To process the sprot XML files into FASTA files, run the following command (on lambda10):
```bash
nohup python protein_search/xml_to_fasta.py --input_dir /nfs/ml_lab/projects/ml_lab/afreiburger/proteins/Uniprot/uniprot/sprot --output_dir data/sprot --num_workers 10 --chunk_size 100 &> sprot.log &
```

To process the trembl XML files into FASTA files, run the following command (on lambda10):
```bash
nohup python protein_search/xml_to_fasta.py --input_dir /nfs/ml_lab/projects/ml_lab/afreiburger/proteins/Uniprot/uniprot/trembl --output_dir data/trembl --num_workers 20 --chunk_size 100 &> trembl.log &
```

## Contributing

For development, it is recommended to use a virtual environment. The following commands will create a virtual environment, install the package in editable mode, and install the pre-commit hooks.
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
