"""Utility to convert a UniProt XML file to a FASTA file."""
from __future__ import annotations
from protein_search.utils import ArgumentsBase, Sequence, write_fasta
from pathlib import Path
from dataclasses import dataclass, field
from lxml import etree
from typing import TypeVar
import functools
from uuid import uuid4
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

T = TypeVar("T")


def parse_uniprot_xml(xml_file: Path) -> list[Sequence]:
    # Load the XML file
    tree = etree.parse(str(xml_file))
    root = tree.getroot()

    # Define UniProt namespace
    ns = {"uni": "http://uniprot.org/uniprot"}

    # Parse the XML file and return a list of sequences with their accession ids
    return [
        Sequence(
            sequence=entry.findtext(".//uni:sequence", namespaces=ns),
            tag=entry.findtext(".//uni:accession", namespaces=ns),
        )
        for entry in root.iterfind(".//uni:entry", namespaces=ns)
    ]


def process_uniprot_xml_files(xml_files: list[Path], output_dir: Path) -> None:
    # Parse the XML files into a list of sequences
    sequences: list[Sequence] = []
    for xml_file in xml_files:
        sequences.extend(parse_uniprot_xml(xml_file))

    # Write the sequences to a FASTA file with a unique name
    write_fasta(sequences, output_dir / f"{uuid4()}.fasta")


@dataclass
class Arguments(ArgumentsBase):
    input_dir: Path = field(
        metadata={"help": "A directory containing XML file with protein sequences"},
    )
    output_dir: Path = field(
        metadata={"help": "An output FASTA file containing protein sequences"},
    )
    num_workers: int = field(
        default=1,
        metadata={
            "help": "Number of worker processes to use for parsing the XML files"
        },
    )
    chunk_size: int = field(
        default=1,
        metadata={"help": "Number of XML files to process in each worker process"},
    )


def batch_data(data: list[T], chunk_size: int) -> list[list[T]]:
    """Batch data into chunks of size chunk_size."""
    batches = [
        data[i * chunk_size : (i + 1) * chunk_size]
        for i in range(0, len(data) // chunk_size)
    ]
    if len(data) > chunk_size * len(batches):
        batches.append(data[len(batches) * chunk_size :])
    return batches


if __name__ == "__main__":
    # Parse arguments from the command line
    args = Arguments.from_cli()

    # Make the output directory
    args.output_dir.mkdir(exist_ok=True)

    # Chunk the input files
    input_files = list(args.input_dir.glob("*.xml"))
    chunks = batch_data(input_files, args.chunk_size)

    # Print some information about the job
    print(f"Found {len(input_files)} XML files in {args.input_dir.resolve()}...")
    print(f"Processing XML files in {len(chunks)} chunks...")
    print(f"Using {args.num_workers} worker processes...")
    print(f"Saving FASTA files to {args.output_dir.resolve()}...")
    print(f"Last chunk size: {len(chunks[-1])}, other chunk sizes: {len(chunks[0])}")

    # Define a worker function that processes a chunk of XML files
    worker_fn = functools.partial(process_uniprot_xml_files, output_dir=args.output_dir)

    # Use a multiprocessing pool to process chunks of XML files in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for _ in tqdm(pool.map(worker_fn, chunks), total=len(chunks)):
            pass
