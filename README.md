# protein_search
Semantic similarity search for proteins.

## Installation
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
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