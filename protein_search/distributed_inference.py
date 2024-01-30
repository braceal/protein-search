from __future__ import annotations
from accelerate import PartialState
from transformers import EsmForMaskedLM, EsmTokenizer, PreTrainedModel, BatchEncoding
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from protein_search.utils import ArgumentsBase, read_fasta


# TODO: For big models, see here: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
# Documentation on using accelerate for inference: https://huggingface.co/docs/accelerate/usage_guides/distributed_inference
# TODO: Skip the for loop over the sequence lenghts using the attention mask:
#   https://stackoverflow.com/questions/65083581/how-to-compute-mean-max-of-huggingface-transformers-bert-token-embeddings-with-a


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return " ".join(self.sequences[idx].upper())


class SequenceCollator:
    def __init__(self, tokenizer: EsmTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")


@torch.no_grad()
def compute_avg_embeddings(
    model: PreTrainedModel, dataloader: DataLoader
) -> np.ndarray:
    """Function to compute averaged hidden embeddings."""
    embeddings = []
    for batch in tqdm(dataloader):
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


if __name__ == "__main__":
    # Parse arguments from the command line
    args = Arguments.from_cli()

    # Initialize distributed state
    distributed_state = PartialState()

    # Load model and tokenizer
    tokenizer = EsmTokenizer.from_pretrained(args.model)
    model = EsmForMaskedLM.from_pretrained(args.model)
    model.eval()
    model.to(distributed_state.device)
    model.compile()

    # Collect all sequence files
    input_files = list(args.input_dir.glob("*"))

    # Make the output directory
    args.output_dir.mkdir(exist_ok=True)

    # Distribute the input files across processes
    with distributed_state.split_between_processes(input_files) as files:
        files: list[Path]  # Spliting the list removes the type information

        # Loop over the files assigned to this process
        for file in tqdm(files, desc="Processing files"):
            # Read fasta file sequences into a list
            sequences = [seq.sequence for seq in read_fasta(file)]

            # Build a torch dataset for efficient batching
            dataloader = DataLoader(
                num_workers=args.num_data_workers,
                pin_memory=True,
                batch_size=args.batch_size,
                dataset=SequenceDataset(sequences),
                collate_fn=SequenceCollator(tokenizer),
            )

            # Compute averaged hidden embeddings
            avg_embeddings = compute_avg_embeddings(model, dataloader)

            # Save or use the averaged embeddings as needed
            # For example, you can save the embeddings to a file
            np.save(args.output_dir / f"{file.name}-embeddings.npy", avg_embeddings)
