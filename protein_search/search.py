"""Search for similar sequences in a dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import faiss
import numpy as np
from datasets import Dataset

from protein_search.distributed_inference import embed_file
from protein_search.distributed_inference import fasta_data_reader
from protein_search.distributed_inference import get_esm_model
from protein_search.utils import read_fasta


def generate_dataset(
    fasta_files: list[Path],
    embedding_files: list[Path],
) -> Iterator[dict[str, str | np.ndarray]]:
    """Generate a dataset from the FASTA and embedding files.

    Parameters
    ----------
    fasta_files : list[Path]
        The list of FASTA files.
    embedding_files : list[Path]
        The list of embedding files.

    Yields
    ------
    dict[str, str | np.ndarray]
        The dictionary containing the sequence tag and embeddings.
    """
    for fasta_file, embedding_file in zip(fasta_files, embedding_files):
        # Read the FASTA file and embeddings one by one
        sequences = read_fasta(fasta_file)
        embeddings = np.load(embedding_file)
        # Yield the sequences and embeddings for the given data files
        for sequence, embedding in zip(sequences, embeddings):
            yield {'tags': sequence.tag, 'embeddings': embedding}


def build_faiss_index(
    fasta_dir: Path,
    embedding_dir: Path,
    dataset_dir: Path,
    pattern: str = '*.fasta',
) -> None:
    """Build a FAISS index for a dataset.

    Parameters
    ----------
    fasta_dir : Path
        The directory containing the FASTA files.
    embedding_dir : Path
        The directory containing the embeddings .npy files
        (with the same name as FASTA files).
    dataset_dir : Path
        The path of the output dataset directory to write.
    pattern : str
        The pattern to match the FASTA files.
    """
    # Create the dataset directory if it does not exist
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of FASTA and embedding files
    # Note: We are assuming that the fasta and embedding
    # files have the same name
    fasta_files = list(fasta_dir.glob(pattern))
    embedding_files = [embedding_dir / f'{f.stem}.npy' for f in fasta_files]

    # Create a dataset from the generator
    dataset = Dataset.from_generator(
        generate_dataset,
        gen_kwargs={
            'fasta_files': fasta_files,
            'embedding_files': embedding_files,
        },
    )

    # Build the FAISS index
    dataset.add_faiss_index(
        column='embeddings',
        index_name='embeddings',
        faiss_verbose=True,
        metric_type=faiss.METRIC_L2,
    )

    # Save the dataset to disk
    dataset.save_faiss_index('embeddings', dataset_dir.with_suffix('.index'))
    dataset.drop_index('embeddings')
    dataset.save_to_disk(dataset_dir)


def search(  # noqa: PLR0913
    dataset_dir: Path,
    query_file: Path,
    model_id: str,
    top_k: int = 5,
    batch_size: int = 8,
    num_data_workers: int = 4,
) -> tuple[list[list[float]], list[list[int]]]:
    """Search for similar sequences.

    Parameters
    ----------
    dataset_dir : Path
        The path to the dataset directory.
    query_file : Path
        The query sequence fasta file.
    model_id : str
        The model to use for generating the embeddings.
    top_k : int
        The number of top results to return.
    batch_size : int
        The batch size for computing embeddings of the query sequences.
    num_data_workers : int
        The number of data workers to use for computing embeddings.

    Returns
    -------
    tuple[list[list[float]], list[list[int]]]
        The total scores and indices of the top k similar sequences
        for each query sequence.
    """
    # Load the dataset from disk
    dataset = Dataset.load_from_disk(dataset_dir)
    dataset.load_faiss_index('embeddings', dataset_dir.with_suffix('.index'))

    # Get the embeddings for the query sequences
    query_embeddings = embed_file(
        file=query_file,
        model_id=model_id,
        batch_size=batch_size,
        num_data_workers=num_data_workers,
        data_reader_fn=fasta_data_reader,
        model_fn=get_esm_model,
    )

    # Convert the query embeddings to float32 for FAISS
    query_embeddings = query_embeddings.astype(np.float32)

    # Search the dataset for the top k similar sequences
    out = dataset.search_batch(
        index_name='embeddings',
        queries=query_embeddings,
        k=top_k,
    )

    # Return the total scores and indices
    return out.total_scores, out.total_indices
