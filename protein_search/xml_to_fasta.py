from bs4 import BeautifulSoup
from cli import ArgumentsBase
from pathlib import Path
from dataclasses import dataclass, field


def parse_uniprot_xml(xml_file: Path):
    with open(xml_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "xml")

    proteins = []
    # TODO: turn into list comprehension
    # TODO: Create sequence dataclass here and return a list of sequences
    for entry in soup.find_all("entry"):
        protein = {}

        # Get protein ID
        protein["id"] = entry.accession.text

        # Get protein sequence
        protein["sequence"] = entry.sequence

        # You can add more fields as needed
        proteins.append(protein)

    return proteins


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
    proteins = parse_uniprot_xml(args.input_xml)

    for protein in proteins:
        print(f"Protein ID: {protein['id']}")
        print(f"Protein Sequence: {protein.get('sequence', 'N/A')}")
        print("-" * 30)
