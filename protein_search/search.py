"""Search for similar sequences in a dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import faiss
import numpy as np
import torch
from datasets import Dataset
from datasets.search import BatchedSearchResults

from protein_search.distributed_inference import average_pool
from protein_search.embedders import BaseEmbedder
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
    faiss_index_name: str = 'embeddings',
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
    faiss_index_name : str
        The name of the dataset field containing the FAISS index,
        by default 'embeddings'.
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
        column=faiss_index_name,
        index_name=faiss_index_name,
        faiss_verbose=True,
        metric_type=faiss.METRIC_L2,
    )

    # Save the dataset to disk
    dataset.save_faiss_index(
        faiss_index_name,
        dataset_dir.with_suffix('.index'),
    )
    dataset.drop_index(faiss_index_name)
    dataset.save_to_disk(dataset_dir)


# def search(
#     dataset_dir: Path,
#     query_file: Path,
#     model_id: str,
#     top_k: int = 5,
#     batch_size: int = 8,
#     num_data_workers: int = 4,
# ) -> tuple[list[list[float]], list[list[int]]]:
#     """Search for similar sequences.

#     Parameters
#     ----------
#     dataset_dir : Path
#         The path to the dataset directory.
#     query_file : Path
#         The query sequence fasta file.
#     model_id : str
#         The model to use for generating the embeddings.
#     top_k : int
#         The number of top results to return.
#     batch_size : int
#         The batch size for computing embeddings of the query sequences.
#     num_data_workers : int
#         The number of data workers to use for computing embeddings.

#     Returns
#     -------
#     tuple[list[list[float]], list[list[int]]]
#         The total scores and indices of the top k similar sequences
#         for each query sequence.
#     """
#     # Load the dataset from disk
#     dataset = Dataset.load_from_disk(dataset_dir)
#     dataset.load_faiss_index('embeddings', dataset_dir.with_suffix('.index'))

#     # Get the embeddings for the query sequences
#     query_embeddings = embed_file(
#         file=query_file,
#         model_id=model_id,
#         batch_size=batch_size,
#         num_data_workers=num_data_workers,
#         data_reader_fn=fasta_data_reader,
#         model_fn=get_esm_model,
#     )

#     # Convert the query embeddings to float32 for FAISS
#     query_embeddings = query_embeddings.astype(np.float32)

#     # Search the dataset for the top k similar sequences
#     out = dataset.search_batch(
#         index_name='embeddings',
#         queries=query_embeddings,
#         k=top_k,
#     )

#     # Return the total scores and indices
#     return out.total_scores, out.total_indices


class SimilaritySearch:
    """Similarity search class for searching similar sequences in a dataset."""

    def __init__(
        self,
        dataset_dir: Path,
        embedder: BaseEmbedder,
        faiss_index_file: Path | None = None,
        faiss_index_name: str = 'embeddings',
    ) -> None:
        """Initialize the SimilaritySearch class.

        Parameters
        ----------
        dataset_dir : Path
            The path to the dataset directory.
        faiss_index_file : Path, optional
            The path to the FAISS index file, by default None,
            in which case the FAISS index file is assumed to be
            in the same directory as the dataset with a .index extension.
        faiss_index_name : str, optional
            The name of the dataset field containing the FAISS index,
            by default 'embeddings'.
        """
        # By default, the FAISS index file has the same name as the dataset
        # and is saved with a .index extension in the directory containing
        # the dataset
        if faiss_index_file is None:
            faiss_index_file = dataset_dir.with_suffix('.index')

        # Set the name of the dataset field containing the FAISS index
        self.faiss_index_name = faiss_index_name

        # Load the dataset from disk
        self.dataset = Dataset.load_from_disk(dataset_dir)
        self.dataset.load_faiss_index(self.faiss_index_name, faiss_index_file)

        # Load the model once for generating the embeddings
        self.embedder = embedder

    def search(
        self,
        query_sequence: str | list[str],
        top_k: int = 5,
    ) -> BatchedSearchResults:
        """Search for similar sequences.

        Parameters
        ----------
        query_sequence : str | list[str]
            The single query sequence or list of query sequences.
        top_k : int
            The number of top results to return.

        Returns
        -------
        BatchedSearchResults:
            A namedtuple with list[list[float]] (.total_scores) of scores for
            each  of the top_k returned items and a list[list[int]]]
            (.total_indices) of indices for each of the top_k returned items
            for each query sequence.
        """
        # Embed the query sequences
        query_embeddings = self.get_pooled_embeddings(query_sequence)

        # Search the dataset for the top k similar sequences
        return self.dataset.search_batch(
            index_name=self.faiss_index_name,
            queries=query_embeddings,
            k=top_k,
        )

    @torch.no_grad()
    def get_pooled_embeddings(
        self,
        query_sequence: str | list[str],
    ) -> np.ndarray:
        """Get the embeddings for the query sequences.

        Parameters
        ----------
        query_sequence : str | list[str]
            The single query sequence or list of query sequences.

        Returns
        -------
        np.ndarray
            The embeddings of the query sequences
            (shape: [num_sequences, embedding_size])
        """
        # Convert the query sequence to a list if it is a single sequence
        if isinstance(query_sequence, str):
            query_sequence = [query_sequence]

        # Tokenize the query sequences
        batch_encoding = self.embedder.tokenizer(
            query_sequence,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        # Move the batch encoding to the device
        inputs = batch_encoding.to(self.embedder.device)

        # Embed the query sequences
        query_embeddings = self.embedder.embed(inputs)

        # Compute average embeddings for the query sequences
        pool_embeds = average_pool(query_embeddings, inputs.attention_mask)

        # Convert the query embeddings to numpy float32 for FAISS
        pool_embeds = pool_embeds.cpu().numpy().astype(np.float32)

        return pool_embeds

    def get_sequence_tags(self, indices: list[int]) -> list[str]:
        """Get the sequence tags for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices returned from the search.

        Returns
        -------
        list[str]
            The list of sequence tags.
        """
        return self.dataset['tags'][indices].tolist()
