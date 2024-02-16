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
                yield {'tags': sequence.tag, 'embeddings': embedding}

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

    #breakpoint()
    #index = faiss.index_factory(320, "Flat", faiss.METRIC_INNER_PRODUCT)
    #index = faiss.index_cpu_to_all_gpus(index, useFloat16=True)
    #embs = [emb["embeddings"] in generate_dataset(fasta_files, embedding_files)]
    #index.add(embs)
    #dataset.add_faiss_index("embeddings", custom_index=index)


    # Build the FAISS index
    dataset.add_faiss_index(
        column='embeddings',
        index_name='embeddings',
        faiss_verbose=True,
        metric_type=faiss.METRIC_L2, #faiss.METRIC_INNER_PRODUCT,
        #string_factory="IndexHNSWFlat",
        #dtype=np.float16,
    )
    #breakpoint()
    # Save the dataset to disk
    #dataset.save_faiss_index('embeddings', dataset_dir)
    dataset.save_faiss_index('embeddings', dataset_dir.with_suffix(".index"))
    dataset.drop_index("embeddings")
    dataset.save_to_disk(dataset_dir)


@app.command()
def search(  # noqa: PLR0913
    dataset_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--dataset_dir',
        '-d',
        help='The path to the dataset file.',
    ),
    query_file: Path = typer.Option(  # noqa: B008
        ...,
        '--query_file',
        '-q',
        help='The query sequence fasta file.',
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
    batch_size: int = typer.Option(
        8,
        '--batch_size',
        '-b',
        help='The batch size for computing embeddings of the query sequences.',
    ),
    num_data_workers: int = typer.Option(
        4,
        '--num_data_workers',
        '-w',
        help='The number of data workers to use for computing embeddings.',
    ),
) -> None:
    """Search for similar sequences."""
    from datasets import Dataset
    import numpy as np

    from protein_search.distributed_inference import embed_file
    from protein_search.distributed_inference import fasta_data_reader
    from protein_search.distributed_inference import get_esm_model

    # Load the dataset from disk
    #dataset = Dataset.load_from_disk(dataset_dir.as_posix())
    dataset = Dataset.load_from_disk(dataset_dir)
    dataset.load_faiss_index('embeddings', dataset_dir.with_suffix(".index"))

    # Get the embeddings for the query sequences
    query_embeddings = embed_file(
        file=query_file,
        model_id=model_id,
        batch_size=batch_size,
        num_data_workers=num_data_workers,
        data_reader_fn=fasta_data_reader,
        model_fn=get_esm_model,
    ).astype(np.float32)


    # Perform the search
    #all_scores, all_samples = [], []
    #for query_embedding in query_embeddings:
    #    scores, samples = dataset.get_nearest_examples(
    #        'embeddings',
    #        query_embedding,
    #        k=top_k,
    #    )
    #    all_scores.append(scores)
    #    all_samples.append(samples)

    # Read the query sequences
    # sequences = read_fasta(query_file)
    # raw_sequences = [sequence.sequence for sequence in sequences]
    #breakpoint() 
    #all_scores, all_indices = [], []
    #for query_embedding in query_embeddings:
    #    scores, indices = dataset.search(index_name="embeddings", query=query_embedding.reshape(-1, 1), k=top_k)
    #    all_scores.append(scores)
    #    all_indices.append(indices)

    out = dataset.search_batch(index_name="embeddings", queries=query_embeddings, k=top_k)

    # Print the results
    for score, ind in zip(out.total_scores, out.total_indices):
        print(score, ind)

    #breakpoint()

def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
