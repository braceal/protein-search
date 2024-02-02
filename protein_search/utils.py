"""Utilities for protein_search."""
from __future__ import annotations
import re
from argparse import ArgumentParser
from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path
from typing import Union, TypeVar, Type, get_type_hints
import json
import yaml
from pydantic import BaseModel as _BaseModel


PathLike = Union[str, Path]

T = TypeVar("T")


class BaseModel(_BaseModel):
    """An interface to add JSON/YAML serialization to Pydantic models."""

    def write_json(self, path: PathLike) -> None:
        """Write the model to a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.
        """
        with open(path, "w") as fp:
            json.dump(self.dict(), fp, indent=2)

    @classmethod
    def from_json(cls: type[T], path: PathLike) -> T:
        """Load the model from a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.

        Returns:
        -------
        T
            A specific BaseModel instance.
        """
        with open(path) as fp:
            data = json.load(fp)
        return cls(**data)

    def write_yaml(self, path: PathLike) -> None:
        """Write the model to a YAML file.

        Parameters
        ----------
        path : str
            The path to the YAML file.
        """
        with open(path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: type[T], path: PathLike) -> T:
        """Load the model from a YAML file.

        Parameters
        ----------
        path : PathLike
            The path to the YAML file.

        Returns:
        -------
        T
            A specific BaseModel instance.
        """
        with open(path) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class ArgumentsBase:
    """Base class for parsing arguments from the command line."""

    @classmethod
    def from_cli(cls: Type[T]) -> T:
        parser = ArgumentParser()

        # Parse the type hints for the dataclass, this is used to set
        # the type of each argument. Simply passing f.type to the parser
        # will not work for Path objects, since the dataclass Path type
        # is not recognized as a callable by the parser.
        type_hints = get_type_hints(cls)

        # Add arguments for each field in the dataclass
        for f in fields(cls):
            # Set up the keyword arguments for the parser
            kwargs = {
                "type": type_hints[f.name],
                "required": f.default == MISSING,
                "help": f.metadata.get("help", ""),
            }
            # Use the default value if the field is not required
            if not kwargs["required"]:
                kwargs["default"] = f.default

            # Add the argument to the parser
            parser.add_argument(f"--{f.name}", **kwargs)

        args = parser.parse_args()
        return cls(**vars(args))


@dataclass
class Sequence:
    sequence: str
    """Biological sequence (Nucleotide/Amino acid sequence)."""
    tag: str
    """Sequence description tag."""


def read_fasta(fasta_file: PathLike) -> list[Sequence]:
    """Reads fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)
    ]

    return [
        Sequence(sequence=seq, tag=tag) for seq, tag in zip(lines[1::2], lines[::2])
    ]


def write_fasta(
    sequences: Sequence | list[Sequence], fasta_file: PathLike, mode: str = "w"
) -> None:
    """Write or append sequences to a fasta file."""
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        for seq in seqs:
            f.write(f">{seq.tag}\n{seq.sequence}\n")


if __name__ == "__main__":

    @dataclass
    class Arguments(ArgumentsBase):
        input_dir: Path = field(
            metadata={"help": "An input directory containing .fasta files"},
        )
        output_dir: Path = field(
            metadata={"help": "An output directory to save the embeddings"},
        )
        model: str = field(
            default="facebook/esm2_t6_8M_UR50D",
            metadata={"help": "Model name or path"},
        )
        batch_size: int = field(default=8, metadata={"help": "Inference batch size"})
        num_data_workers: int = field(
            default=4, metadata={"help": "Number of data workers for batching"}
        )

    args = Arguments.from_cli()
    print(args)

    args2 = Arguments(input_dir=Path("foo"), output_dir=Path("bar"))
    print(args2)
