"""CLI for protein-search."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

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
def build_faiss_index(
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
    output_dataset_file: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dataset_file',
        '-o',
        help='The path of the output dataset file to write.',
    ),
    pattern: str = typer.Option(
        '*.fasta',
        '--pattern',
        '-p',
        help='The pattern to match the FASTA files.',
    ),
) -> None:
    """Build a FAISS index for a dataset."""
    import faiss
    import numpy as np
    from datasets import Dataset

    from protein_search.utils import read_fasta

    def generate_dataset(
        fasta_files: list[Path],
        embedding_files: list[Path],
    ) -> Iterator[dict[str, str | np.ndarray]]:
        """Generate a dataset from the FASTA and embedding files."""
        for fasta_file, embedding_file in zip(fasta_files, embedding_files):
            # Read the FASTA file and embeddings one by one
            sequences = read_fasta(fasta_file)
            embeddings = np.load(embedding_file)
            # Yield the sequences and embeddings for the given data files
            for sequence, embedding in zip(sequences, embeddings):
                yield {'tag': sequence.tag, 'embedding': embedding}

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
        metric_type=faiss.METRIC_INNER_PRODUCT,
        # string_factory=,
    )

    # Save the dataset to disk
    dataset.save_to_disk(output_dataset_file)


@app.command()
def search(
    dataset_file: Path = typer.Option(  # noqa: B008
        ...,
        '--dataset_file',
        '-d',
        help='The path to the dataset file.',
    ),
    query_sequence: str = typer.Option(
        ...,
        '--query_sequence',
        '-q',
        help='The query sequence.',
    ),
    model_id: str = typer.Option(
        'facebook/esm2_t6_8M_UR50D',
        '--model_id',
        '-m',
        help='The model to use for generating the embeddings.',
    ),
    top_k: int = typer.Option(
        5,
        '--top_k',
        '-k',
        help='The number of top results to return.',
    ),
) -> None:
    """Search for similar sequences."""
    import torch
    from datasets import Dataset

    from protein_search.distributed_inference import average_pool
    from protein_search.distributed_inference import get_esm_model

    # Load the dataset from disk
    dataset = Dataset.load_from_disk(dataset_file.as_posix())

    # Get the ESM model
    model, tokenizer = get_esm_model(model_id)  # type: ignore[misc]

    # Tokenize the query sequence
    batch_encoding = tokenizer(query_sequence, return_tensors='pt')

    # Get the query embedding
    with torch.no_grad():
        # Move the inputs to the device
        inputs = batch_encoding.to(model.device)

        # Get the model outputs with a forward pass
        outputs = model(**inputs, output_hidden_states=True)

        # Get the last hidden states
        last_hidden_state = outputs.hidden_states[-1]

        # Get the mean of the last hidden states
        query_embedding = average_pool(
            last_hidden_state,
            inputs.attention_mask,
        )

    # Perform the search
    scores, samples = dataset.get_nearest_examples(
        'embeddings',
        query_embedding,
        k=top_k,
    )

    # Print the results
    for score, sample in zip(scores, samples):
        print(score, sample)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
