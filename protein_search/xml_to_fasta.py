"""Utility to convert a UniProt XML file to a FASTA file."""
from __future__ import annotations
from bs4 import BeautifulSoup
from protein_search.utils import ArgumentsBase, Sequence, write_fasta
from pathlib import Path
from dataclasses import dataclass, field
from lxml import etree


def parse_uniprot_xml(xml_file: Path) -> list[Sequence]:
    tree = etree.parse(str(xml_file))
    root = tree.getroot()

    # Define UniProt namespace
    ns = {'uni': 'http://uniprot.org/uniprot'}

    return [
        Sequence(
            sequence=entry.findtext(".//uni:sequence", namespaces=ns), # Get protein sequence
            tag=entry.findtext(".//uni:accession", namespaces=ns),  # Get protein ID
        )
        for entry in root.iterfind(".//uni:entry", namespaces=ns)
    ]


@dataclass
class Arguments(ArgumentsBase):
    input_xml: Path = field(
        metadata={"help": "An input XML file containing protein information"},
    )
    output_fasta: Path = field(
        metadata={"help": "An output FASTA file containing protein sequences"},
    )

def speed_test():
    xml_files = list(Path("/nfs/ml_lab/projects/ml_lab/afreiburger/proteins/Uniprot/uniprot/trembl").glob("block_1000*"))
    from tqdm import tqdm
    sequences = []
    print(xml_files)
    for xml_file in tqdm(xml_files):
        seqs = parse_uniprot_xml(xml_file)
        print(f"Found {len(seqs)} sequences in {xml_file}")
        sequences.extend(seqs)

    write_fasta(sequences, "block_1000_test_v2.fasta")

if __name__ == "__main__":
    speed_test()
    exit(0)

    # Parse arguments from the command line
    args = Arguments.from_cli()

    # Parse the XML file
    sequences = parse_uniprot_xml(args.input_xml)

    # Log the number of sequences
    print(
        f"Found {len(sequences)} sequences in {args.input_xml}, "
        f" writing to {args.output_fasta}"
    )

    # Write the FASTA file
    write_fasta(sequences, args.output_fasta)
