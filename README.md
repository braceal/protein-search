# protein-search
Semantic similarity search for proteins.

## Installation

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

## Usage

To process the sprot XML files into FASTA files, run the following command (on lambda10):
```bash
nohup python protein_search/xml_to_fasta.py --input_dir /nfs/ml_lab/projects/ml_lab/afreiburger/proteins/Uniprot/uniprot/sprot --output_dir data/sprot --num_workers 10 --chunk_size 100 &> sprot.log &
```

To process the trembl XML files into FASTA files, run the following command (on lambda10):
```bash
nohup python protein_search/xml_to_fasta.py --input_dir /nfs/ml_lab/projects/ml_lab/afreiburger/proteins/Uniprot/uniprot/trembl --output_dir data/trembl --num_workers 20 --chunk_size 100 &> trembl.log &
```
