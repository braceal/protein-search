from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path
from typing import TypeVar, Type, get_type_hints

T = TypeVar("T")


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
