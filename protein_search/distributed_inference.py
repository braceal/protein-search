from __future__ import annotations
from accelerate import PartialState
from transformers import EsmForMaskedLM, EsmTokenizer, PreTrainedModel
import re
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, fields, MISSING
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from transformers import BatchEncoding, DataCollatorForLanguageModeling
from typing import TypeVar, Type, get_type_hints

T = TypeVar("T")


# TODO: For big models, see here: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
# Documentation on using accelerate for inference: https://huggingface.co/docs/accelerate/usage_guides/distributed_inference
# TODO: Skip the for loop over the sequence lenghts using the attention mask:
#   https://stackoverflow.com/questions/65083581/how-to-compute-mean-max-of-huggingface-transformers-bert-token-embeddings-with-a


class ArgumentBase:
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


def read_fasta(fasta_file: Path) -> list[str]:
    """Reads fasta file sequences and returns a list of sequences
    compatible with the tokenizer."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)
    ]

    return [" ".join(seq).upper() for seq in lines[1::2]]


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str], tokenizer: EsmTokenizer) -> None:
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> BatchEncoding:
        return self.tokenizer(self.sequences[idx], return_tensors="pt")


@torch.no_grad()
def compute_avg_embeddings(
    model: PreTrainedModel, dataloader: DataLoader
) -> np.ndarray:
    """Function to compute averaged hidden embeddings."""
    embeddings = []
    for batch in dataloader:
        batch = batch.to(model.device)
        outputs = model(**batch, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        seq_lengths = batch.attention_mask.sum(axis=1)
        for seq_len, elem in zip(seq_lengths, last_hidden_states):
            # Compute averaged embedding
            embedding = elem[1 : seq_len - 1, :].mean(dim=0).cpu().numpy()
            embeddings.append(embedding)

    return np.array(embeddings)


@dataclass
class Arguments(ArgumentBase):
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


if __name__ == "__main__":
    # Parse arguments from the command line
    args = Arguments.from_cli()

    # Initialize distributed state singleton
    distributed_state = PartialState()

    # Load model and tokenizer
    tokenizer = EsmTokenizer.from_pretrained(args.model)
    model = EsmForMaskedLM.from_pretrained(args.model)
    model.eval().to(distributed_state.device)

    # Collect all sequence files
    input_files = list(args.input_dir.glob("*"))

    # Distribute the input files across processes
    with distributed_state.split_between_processes(input_files) as files:
        files: list[Path]  # Spliting the list removes the type information

        # Loop over the files assigned to this process
        for file in files:
            # Read fasta file
            sequences = read_fasta(file)

            # Build a torch dataset for efficient batching
            dataloader = DataLoader(
                num_workers=4,
                pin_memory=True,
                batch_size=args.batch_size,
                dataset=SequenceDataset(sequences, tokenizer),
                collate_fn=DataCollatorForLanguageModeling(tokenizer),
            )

            # Compute averaged hidden embeddings
            avg_embeddings = compute_avg_embeddings(model, dataloader)

            # Save or use the averaged embeddings as needed
            # For example, you can save the embeddings to a file
            np.save(f"{file.name}-embeddings.npy", avg_embeddings)
