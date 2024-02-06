"""Distributed inference for generating embeddings."""

from __future__ import annotations

import functools
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BatchEncoding
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from protein_search.parsl import ComputeSettingsTypes
from protein_search.registry import register
from protein_search.utils import BaseModel

# TODO: For big models, see here: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
# Documentation on using accelerate for inference: https://huggingface.co/docs/accelerate/usage_guides/distributed_inference
# TODO: Skip the for loop over the sequence lengths using the attention mask:
#   https://stackoverflow.com/questions/65083581/how-to-compute-mean-max-of-huggingface-transformers-bert-token-embeddings-with-a


class InMemoryDataset(Dataset):
    """Holds the data in memory for efficient batching."""

    def __init__(self, data: list[str]) -> None:
        self.data = data

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        """Get an item from the dataset."""
        return self.data[idx]


class DataCollator:
    """Data collator for batching sequences."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """Initialize the data collator."""
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]) -> BatchEncoding:
        """Collate the batch of sequences."""
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )


@torch.no_grad()
def compute_avg_embeddings(
    model: PreTrainedModel,
    dataloader: DataLoader,
) -> np.ndarray:
    """Compute averaged hidden embeddings.

    Parameters
    ----------
    model : PreTrainedModel
        The model to use for inference.
    dataloader : DataLoader
        The dataloader to use for batching the data.

    Returns
    -------
    np.ndarray
        A numpy array of averaged hidden embeddings.
    """
    import numpy as np
    from tqdm import tqdm

    # TODO: Instead of using a list, store the embeddings in a torch tensor
    # with the size reserved for the entire dataset.
    embeddings = []
    for batch in tqdm(dataloader):
        inputs = batch.to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        seq_lengths = inputs.attention_mask.sum(axis=1)
        for seq_len, elem in zip(seq_lengths, last_hidden_states):
            # Compute averaged embedding
            embedding = elem[1 : seq_len - 1, :].mean(dim=0).cpu().numpy()
            embeddings.append(embedding)

    return np.array(embeddings)


@torch.no_grad()
def compute_avg_embeddings_v2(
    model: PreTrainedModel,
    dataloader: DataLoader,
) -> np.ndarray:
    """Compute averaged hidden embeddings.

    Parameters
    ----------
    model : PreTrainedModel
        The model to use for inference.
    dataloader : DataLoader
        The dataloader to use for batching the data.

    Returns
    -------
    np.ndarray
        A numpy array of averaged hidden embeddings.
    """
    from tqdm import tqdm

    device = model.device

    # Get the number of embeddings and the embedding size
    num_embeddings = len(dataloader.dataset)
    embedding_size = model.config.hidden_size

    # Initialize a torch tensor for storing embeddings on the GPU
    embeddings = torch.empty(
        (num_embeddings, embedding_size),
        dtype=model.dtype,
        device=device,
    )

    # Index for storing embeddings
    idx = 0

    # Iterate over batches
    for batch in tqdm(dataloader):
        # Move batch to the device
        batch = batch.to(device)  # noqa: PLW2901

        # Forward pass of the model
        outputs = model(**batch, output_hidden_states=True)

        # Get the last hidden states
        hidden_states = outputs.hidden_states[-1]

        # Compute mask for valid positions
        mask = (
            batch.attention_mask[:, 1:-1]
            .unsqueeze(-1)
            .expand_as(hidden_states)
        )

        # Sum over valid positions and divide by the number of valid positions
        sum_embeddings = torch.sum(hidden_states[:, 1:-1, :] * mask, dim=1)

        # Divide by the number of valid positions
        avg_embeddings = sum_embeddings / mask.sum(dim=1).unsqueeze(-1)

        # Move embeddings to CPU before storing
        embeddings[idx : idx + batch.size(0), :] = avg_embeddings.cpu()
        idx += batch.size(0)

    return embeddings.numpy()


def embed_file(  # noqa: PLR0913
    file: Path,
    output_dir: Path,
    model_id: str,
    batch_size: int,
    num_data_workers: int,
    data_reader_fn: Callable[[Path], list[str]],
    model_fn: Callable[[str], tuple[PreTrainedModel, PreTrainedTokenizer]],
) -> None:
    """Embed a single file and save a numpy array with embeddings."""
    # Imports are here since this function is called in a parsl process
    import numpy as np
    from torch.utils.data import DataLoader

    from protein_search.distributed_inference import compute_avg_embeddings
    from protein_search.distributed_inference import DataCollator
    from protein_search.distributed_inference import InMemoryDataset

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
    np.save(output_dir / f'{file.name}-embeddings.npy', avg_embeddings)


@register()  # type: ignore[arg-type]
def get_esm_model(
    model_id: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Initialize the model and tokenizer."""
    # Subsequent calls will be warmstarts.
    import torch
    from transformers import EsmForMaskedLM
    from transformers import EsmTokenizer

    # Load model and tokenizer
    tokenizer = EsmTokenizer.from_pretrained(model_id)
    model = EsmForMaskedLM.from_pretrained(model_id)

    # Convert the model to half precision
    model.half()

    # Set the model to evaluation mode
    model.eval()

    # Load the model onto the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Compile the model for faster inference
    model = torch.compile(model, fullgraph=True)

    return model, tokenizer


def fasta_data_reader(data_file: Path) -> list[str]:
    """Read a fasta file and return a list of sequences."""
    from protein_search.utils import read_fasta

    return [' '.join(seq.sequence.upper()) for seq in read_fasta(data_file)]


class Config(BaseModel):
    """Configuration for distributed inference."""

    # An input directory containing .fasta files.
    input_dir: Path
    # An output directory to save the embeddings.
    output_dir: Path
    # A set of glob patterns to match the input files.
    glob_files: list[str] = Field(default=['*'])
    # Model name or path.
    model: str = 'facebook/esm2_t6_8M_UR50D'
    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Settings for the parsl compute backend.
    compute_settings: ComputeSettingsTypes


if __name__ == '__main__':
    # Parse arguments from the command line
    parser = ArgumentParser(description='Embed protein sequences using ESM')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the .yaml configuration file',
    )
    args = parser.parse_args()

    # Load the configuration
    config = Config.from_yaml(args.config)

    # Collect all input files
    input_files = []
    for pattern in config.glob_files:
        input_files.extend(list(config.input_dir.glob(pattern)))

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
    parsl_config = config.compute_settings.get_config(
        config.output_dir / 'parsl',
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        pool.map(worker_fn, input_files)
