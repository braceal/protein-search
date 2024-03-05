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

METRICS = {
    'l2': faiss.METRIC_L2,
    'inner_product': faiss.METRIC_INNER_PRODUCT,
}


def generate_dataset(
    fasta_files: list[Path],
    embedding_files: list[Path],
    metric: str = 'inner_product',
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

        if metric == 'inner_product':
            # Normalize the embeddings for inner product search
            # to make the cosine similarity equivalent to inner product. See:
            # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)

        # Yield the sequences and embeddings for the given data files
        for sequence, embedding in zip(sequences, embeddings):
            yield {'tags': sequence.tag, 'embeddings': embedding}


def build_faiss_index(  # noqa: PLR0913
    fasta_dir: Path,
    embedding_dir: Path,
    dataset_dir: Path,
    pattern: str = '*.fasta',
    faiss_index_name: str = 'embeddings',
    metric: str = 'inner_product',
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
    metric : str
        The metric to use for the FAISS index ['l2', 'inner_product'],
        by default 'inner_product'.
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
            'metric': metric,
        },
    )

    # Build the FAISS index
    dataset.add_faiss_index(
        column=faiss_index_name,
        index_name=faiss_index_name,
        faiss_verbose=True,
        metric_type=METRICS[metric],
    )

    # Save the dataset to disk
    dataset.save_faiss_index(
        faiss_index_name,
        dataset_dir.with_suffix('.index'),
    )
    dataset.drop_index(faiss_index_name)
    dataset.save_to_disk(dataset_dir)


class SimilaritySearch:
    """Similarity search class for searching similar sequences in a dataset."""

    def __init__(  # noqa: PLR0913
        self,
        dataset_dir: Path,
        embedder: BaseEmbedder,
        faiss_index_file: Path | None = None,
        faiss_index_name: str = 'embeddings',
        metric: str = 'inner_product',
    ) -> None:
        """Initialize the SimilaritySearch class.

        Parameters
        ----------
        dataset_dir : Path
            The path to the dataset directory.
        embedder : BaseEmbedder
            The embedder instance to use for embedding sequences.
        faiss_index_file : Path, optional
            The path to the FAISS index file, by default None,
            in which case the FAISS index file is assumed to be
            in the same directory as the dataset with a .index extension.
        faiss_index_name : str, optional
            The name of the dataset field containing the FAISS index,
            by default 'embeddings'.
        metric : str, optional
            The metric to use for the FAISS index ['l2', 'inner_product'],
            by default 'inner_product'.
        """
        self.faiss_index_name = faiss_index_name
        self.metric = metric
        self.embedder = embedder

        # By default, the FAISS index file has the same name as the dataset
        # and is saved with a .index extension in the directory containing
        # the dataset
        if faiss_index_file is None:
            faiss_index_file = dataset_dir.with_suffix('.index')

        # Load the dataset from disk
        self.dataset = Dataset.load_from_disk(dataset_dir)
        self.dataset.load_faiss_index(self.faiss_index_name, faiss_index_file)

    def search(
        self,
        query_sequence: str | list[str] | None = None,
        query_embedding: np.ndarray | None = None,
        top_k: int = 1,
    ) -> BatchedSearchResults:
        """Search for similar sequences.

        Parameters
        ----------
        query_sequence : str | list[str]
            The single query sequence or list of query sequences.
        top_k : int
            The number of top results to return, by default 1.

        Returns
        -------
        BatchedSearchResults:
            A namedtuple with list[list[float]] (.total_scores) of scores for
            each  of the top_k returned items and a list[list[int]]]
            (.total_indices) of indices for each of the top_k returned items
            for each query sequence.

        Raises
        ------
        ValueError
            If both query_sequence and query_embedding are None.
        """
        # Check whether arguments are valid
        if query_sequence is None and query_embedding is None:
            raise ValueError(
                'Provide at least one of query_sequence or query_embedding.',
            )

        # Embed the query sequences
        if query_embedding is None:
            assert query_sequence is not None
            query_embedding = self.get_pooled_embeddings(query_sequence)

        # Search the dataset for the top k similar sequences
        return self.dataset.search_batch(
            index_name=self.faiss_index_name,
            queries=query_embedding,
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

        # Normalize the embeddings for inner product search
        if self.metric == 'inner_product':
            faiss.normalize_L2(pool_embeds)

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
        return [self.dataset['tags'][i] for i in indices]
