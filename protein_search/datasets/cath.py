"""CATH dataset."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


def download_dataset(url: str, filename: Path) -> None:
    """Download a dataset from a URL to a file if it does not exist.

    Parameters
    ----------
    url : str
        The URL of the dataset.
    filename : Path
        The file to save the dataset to.
    """
    if not filename.is_file():
        print(f'Downloading {url} to {filename}')
        urlretrieve(url, filename)


def download_cath_dataset(output_dir: Path) -> None:
    """Download the CATH dataset.

    Parameters
    ----------
    output_dir : Path
        The directory to save the dataset to.
    """
    # Create the data directory if it does not exist
    data_dir = output_dir / 'cath'
    data_dir.mkdir(exist_ok=True)

    # The CATH dataset is available at the following URL prefix
    pf = 'ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_2_0/'

    # Download the CATH non-redundant fasta dataset
    url = 'non-redundant-data-sets/cath-dataset-nonredundant-S20-v4_2_0.fa'
    download_dataset(pf + url, data_dir / 'cath-20.fasta')

    # Download the CATH domain list
    url = 'cath-classification-data/cath-domain-list-v4_2_0.txt'
    download_dataset(pf + url, data_dir / 'cath-domain-list.txt')
