from __future__ import annotations
from transformers import (
    EsmForMaskedLM,
    EsmTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import parsl
import torch
import numpy as np
import functools
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from parsl.concurrent import ParslPoolExecutor
from protein_search.registry import register
from protein_search.utils import BaseModel, read_fasta
from protein_search.parsl import ComputeSettingsTypes

# TODO: For big models, see here: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
# Documentation on using accelerate for inference: https://huggingface.co/docs/accelerate/usage_guides/distributed_inference
# TODO: Skip the for loop over the sequence lenghts using the attention mask:
#   https://stackoverflow.com/questions/65083581/how-to-compute-mean-max-of-huggingface-transformers-bert-token-embeddings-with-a


class InMemoryDataset(Dataset):
    def __init__(self, data: list[str]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")


@torch.no_grad()
def compute_avg_embeddings(
    model: PreTrainedModel, dataloader: DataLoader
) -> np.ndarray:
    """Function to compute averaged hidden embeddings."""
    # TODO: Instead of using a list, store the embeddings in a torch tensor
    # with the size reserved for the entire dataset.
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


def embed_file(
    file: Path,
    output_dir: Path,
    model_id: str,
    batch_size: int,
    num_data_workers: int,
    data_reader_fn: Callable[[Path], list[str]],
    model_fn: Callable[[str], tuple[PreTrainedModel, PreTrainedTokenizer]],
) -> None:
    """Function to embed a single file and save a numpy array with embeddings."""
    # Initialize the model and tokenizer
    model, tokenizer = model_fn(model_id)

    # Read the data
    data = data_reader_fn(file)

    # Build a torch dataset for efficient batching
    dataloader = DataLoader(
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_data_workers,
        dataset=InMemoryDataset(data),
        collate_fn=DataCollator(tokenizer),
    )

    # Compute averaged hidden embeddings
    avg_embeddings = compute_avg_embeddings(model, dataloader)

    # Save or use the averaged embeddings as needed
    # For example, you can save the embeddings to a file
    np.save(output_dir / f"{file.name}-embeddings.npy", avg_embeddings)


@register()
def get_esm_model(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Initialize the model and tokenizer, subsequent calls will be warmstarts."""
    # Load model and tokenizer
    tokenizer = EsmTokenizer.from_pretrained(model_id)
    model = EsmForMaskedLM.from_pretrained(model_id)

    # Convert the model to half precision
    model.half()

    # Set the model to evaluation mode
    model.eval()

    # Load the model onto the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Compile the model for faster inference
    model = torch.compile(model, fullgraph=True)

    return model, tokenizer


def fasta_data_reader(data_file: Path) -> list[str]:
    return [" ".join(seq.sequence.upper()) for seq in read_fasta(data_file)]


class Config(BaseModel):
    input_dir: Path
    """An input directory containing .fasta files."""
    output_dir: Path
    """An output directory to save the embeddings."""
    model: str = "facebook/esm2_t6_8M_UR50D"
    """Model name or path."""
    num_data_workers: int = 4
    """Number of data workers for batching."""
    batch_size: int = 8
    """Inference batch size."""
    compute_settings: ComputeSettingsTypes
    """Settings for the parsl compute backend."""


if __name__ == "__main__":
    # Parse arguments from the command line
    parser = ArgumentParser(description="Embed protein sequences using ESM")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the .yaml configuration file",
    )
    args = parser.parse_args()

    # Load the configuration
    config = Config.from_yaml(args.config)

    # Collect all sequence files
    input_files = list(config.input_dir.glob("*"))

    # Make the output directory
    config.output_dir.mkdir(exist_ok=True)

    # Set the static arguments of the worker function
    worker_fn = functools.partial(
        embed_file,
        output_dir=config.output_dir,
        model_id=config.model,
        batch_size=config.batch_size,
        num_data_workers=config.num_data_workers,
        data_reader_fn=fasta_data_reader,
        model_fn=get_esm_model,
    )

    # Set the parsl compute settings
    parsl_config = config.compute_settings.get_config(config.output_dir / "parsl")
    parsl.load(parsl_config)

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        pool.map(worker_fn, input_files)
