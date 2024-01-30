"""Utility to convert a UniProt XML file to a FASTA file."""
from __future__ import annotations
from bs4 import BeautifulSoup
from protein_search.utils import ArgumentsBase, Sequence
from pathlib import Path
from dataclasses import dataclass, field


def parse_uniprot_xml(xml_file: Path) -> list[Sequence]:
    with open(xml_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "xml")

    return [
        Sequence(
            sequence=entry.sequence.text,  # Get protein sequence
            tag=entry.accession.text,  # Get protein ID
        )
        for entry in soup.find_all("entry")
    ]


@dataclass
class Arguments(ArgumentsBase):
    input_xml: Path = field(
        metadata={"help": "An input XML file containing protein information"},
    )
    output_fasta: Path = field(
        metadata={"help": "An output FASTA file containing protein sequences"},
    )


if __name__ == "__main__":
    # Parse arguments from the command line
    args = Arguments.from_cli()

    # Parse the XML file
    sequences = parse_uniprot_xml(args.input_xml)

    for seq in sequences:
        print(f"Protein ID: {seq.tag}")
        print(f"Protein Sequence: {seq.sequence}")
        print("-" * 30)
