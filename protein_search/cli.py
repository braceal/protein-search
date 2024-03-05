"""CLI for protein-search."""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def download_dataset(
    dataset: str = typer.Option(
        ...,
        '--dataset',
        '-d',
        help='The name of the dataset to download [cath].',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        'data',  # Default to data directory
        '--output_dir',
        '-o',
        help='The directory to save the dataset to.',
    ),
) -> None:
    """Download a dataset."""
    if dataset == 'cath':
        from protein_search.datasets.cath import download_cath_dataset

        download_cath_dataset(output_dir)


@app.command()
def build_index(
    fasta_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--fasta_dir',
        '-f',
        help='The directory containing the FASTA files.',
    ),
    embedding_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--embedding_dir',
        '-e',
        help='The directory containing the embeddings .npy files '
        '(with the same name as FASTA files).',
    ),
    dataset_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--dataset_dir',
        '-o',
        help='The path of the output dataset directory to write.',
    ),
    pattern: str = typer.Option(
        '*.fasta',
        '--pattern',
        '-p',
        help='The pattern to match the FASTA files.',
    ),
    metric: str = typer.Option(
        'inner_product',
        '--metric',
        '-m',
        help='The metric to use for the FAISS index [l2, inner_product].',
    ),
) -> None:
    """Build a FAISS index for a dataset."""
    from protein_search.search import build_faiss_index

    build_faiss_index(
        fasta_dir=fasta_dir,
        embedding_dir=embedding_dir,
        dataset_dir=dataset_dir,
        pattern=pattern,
        metric=metric,
    )


@app.command()
def search_index(  # noqa: PLR0913
    dataset_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--dataset_dir',
        '-d',
        help='The path to the dataset directory.',
    ),
    query_file: Path = typer.Option(  # noqa: B008
        ...,
        '--query_file',
        '-q',
        help='The query sequence fasta file.',
    ),
    pretrained_model_name_or_path: str = typer.Option(
        'facebook/esm2_t6_8M_UR50D',
        '--pretrained_model_name_or_path',
        '-m',
        help='The model weights to use for generating the embeddings.',
    ),
    model_name: str = typer.Option(
        'esm2',
        '--model_name',
        '-n',
        help='The name of the model architecture to use for '
        ' generating the embeddings.',
    ),
    top_k: int = typer.Option(
        1,
        '--top_k',
        '-k',
        help='The number of top results to return.',
    ),
    metric: str = typer.Option(
        'inner_product',
        '--metric',
        '-m',
        help='The metric to use for the FAISS index [l2, inner_product].',
    ),
) -> None:
    """Search for similar sequences."""
    from protein_search.embedders import get_embedder
    from protein_search.search import SimilaritySearch
    from protein_search.utils import read_fasta

    # Initialize the embedder to use for similarity search
    embedder = get_embedder(
        embedder_kwargs={
            # The name of the model architecture to use
            'name': model_name,
            # The model id to use for generating the embeddings
            'pretrained_model_name_or_path': pretrained_model_name_or_path,
            # Use the model in half precision
            'half_precision': True,
            # Set the model to evaluation mode
            'eval_mode': True,
            # Compile the model for faster inference
            # Note: This can actually slow down the inference
            # if the number of queries is small
            'compile_model': False,
        },
    )

    # Initialize the similarity search
    ss = SimilaritySearch(
        dataset_dir=dataset_dir,
        embedder=embedder,
        metric=metric,
    )

    # Read the input query file
    query_sequences = [seq.sequence for seq in read_fasta(query_file)]

    # Search for similar sequences
    results = ss.search(query_sequences, top_k=top_k)

    # Print the results
    for score, ind in zip(results.total_scores, results.total_indices):
        # Get the sequence tags found by the search
        found_tags = ss.get_sequence_tags(ind)
        print(f'scores: {score}, indices: {ind}, tags: {found_tags}')


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
