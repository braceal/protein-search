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
from pydantic import field_validator
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


def average_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average pool the hidden states using the attention mask.

    Parameters
    ----------
    hidden_states : torch.Tensor
        The hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    # Get the sequence lengths
    seq_lengths = attention_mask.sum(axis=1)

    # Set the attention mask to 0 for start and end tokens
    attention_mask[:, 0] = 0
    attention_mask[:, seq_lengths - 1] = 0

    # Get the hidden shape (B, SeqLen, HiddenDim) and batch size (B)
    mask_shape = hidden_states.shape
    # batch_size = mask_shape[0]

    # Create a mask for the pooling operation
    pool_mask = attention_mask.unsqueeze(-1).expand(mask_shape)
    # Sum the embeddings over the sequence length (use the mask to avoid
    # pad, start, and stop tokens)
    sum_embeds = torch.sum(hidden_states * pool_mask, 1)
    # Avoid division by zero for zero length sequences by clamping
    sum_mask = torch.clamp(pool_mask.sum(1), min=1e-9)
    # Compute mean pooled embeddings for each sequence
    return (sum_embeds / sum_mask).cpu()


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
    import torch
    from tqdm import tqdm

    from protein_search.distributed_inference import average_pool

    # Get the number of embeddings and the embedding size
    num_embeddings = len(dataloader.dataset)
    embedding_size = model.config.hidden_size

    # Initialize a torch tensor for storing embeddings on the GPU
    embeddings = torch.empty(
        (num_embeddings, embedding_size),
        dtype=model.dtype,
    )

    # Index for storing embeddings
    idx = 0

    for batch in tqdm(dataloader):
        # Move the batch to the model device
        inputs = batch.to(model.device)

        # Get the model outputs with a forward pass
        outputs = model(**inputs, output_hidden_states=True)

        # Get the last hidden states
        last_hidden_state = outputs.hidden_states[-1]

        # Compute the average pooled embeddings
        pooled_embeds = average_pool(last_hidden_state, inputs.attention_mask)

        # Get the batch size
        batch_size = inputs.attention_mask.shape[0]

        # Store the pooled embeddings in the output buffer
        embeddings[idx : idx + batch_size, :] = pooled_embeds

        # Increment the output buffer index by the batch size
        idx += batch_size

    return embeddings.numpy()


def embed_file(  # noqa: PLR0913
    file: Path,
    model_id: str,
    batch_size: int,
    num_data_workers: int,
    data_reader_fn: Callable[[Path], list[str]],
    model_fn: Callable[[str], tuple[PreTrainedModel, PreTrainedTokenizer]],
) -> np.ndarray:
    """Embed a single file and return a numpy array with embeddings."""
    # Imports are here since this function is called in a parsl process
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
    return compute_avg_embeddings(model, dataloader)


def embed_and_save_file(  # noqa: PLR0913
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

    from protein_search.distributed_inference import embed_file

    # Embed the file
    embeddings = embed_file(
        file=file,
        model_id=model_id,
        batch_size=batch_size,
        num_data_workers=num_data_workers,
        data_reader_fn=data_reader_fn,
        model_fn=model_fn,
    )

    # Save the embeddings to disk
    np.save(output_dir / f'{file.name}-embeddings.npy', embeddings)


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


@register()  # type: ignore[arg-type]
def get_auto_model(
    model_id: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Initialize the model and tokenizer."""
    import torch
    from transformers import AutoModel
    from transformers import AutoTokenizer

    model = AutoModel.from_pretrained(
        model_id,
        deterministic_eval=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load the model onto the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Compile the model for faster inference
    model = torch.compile(model, fullgraph=True)

    return model, tokenizer


def single_sequence_per_line_data_reader(
    data_file: Path,
    header_lines: int = 1,
) -> list[str]:
    """Read a file with one sequence per line.

    Parameters
    ----------
    data_file : Path
        The file to read.
    header_lines : int, optional (default=1)
        The number of header lines to skip.
    """
    return data_file.read_text().splitlines()[header_lines:]


def fasta_data_reader(data_file: Path) -> list[str]:
    """Read a fasta file and return a list of sequences."""
    from protein_search.utils import read_fasta

    return [' '.join(seq.sequence.upper()) for seq in read_fasta(data_file)]


READER_STRATEGIES = {
    'single_sequence_per_line': single_sequence_per_line_data_reader,
    'fasta': fasta_data_reader,
}

MODEL_STRATEGIES = {  # type: ignore[var-annotated]
    'esm': get_esm_model,
    'auto': get_auto_model,
}


class Config(BaseModel):
    """Configuration for distributed inference."""

    # An input directory containing .fasta files.
    input_dir: Path
    # An output directory to save the embeddings.
    output_dir: Path
    # A set of glob patterns to match the input files.
    glob_patterns: list[str] = Field(default=['*'])
    # Model name or path.
    model: str = 'facebook/esm2_t6_8M_UR50D'
    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Strategy for reading the input files.
    data_reader_fn: str = 'fasta'
    # Strategy for initializing the model and tokenizer.
    embedding_model_fn: str = 'esm'
    # Settings for the parsl compute backend.
    compute_settings: ComputeSettingsTypes

    @field_validator('input_dir', 'output_dir')
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        """Resolve the path to an absolute path."""
        return value.resolve()


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

    # Create a directory for the embeddings
    embedding_dir = config.output_dir / 'embeddings'

    # Make the output directory
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # Log the configuration
    config.write_yaml(config.output_dir / 'config.yaml')

    # Get the data reader function
    data_reader_fn = READER_STRATEGIES.get(config.data_reader_fn, None)
    if data_reader_fn is None:
        raise ValueError(
            f'Invalid data reader function: {config.data_reader_fn}',
        )

    # Get the model function
    model_fn = MODEL_STRATEGIES.get(config.embedding_model_fn, None)
    if model_fn is None:
        raise ValueError(
            f'Invalid model function: {config.embedding_model_fn}',
        )

    # Set the static arguments of the worker function
    worker_fn = functools.partial(
        embed_and_save_file,
        output_dir=embedding_dir,
        model_id=config.model,
        batch_size=config.batch_size,
        num_data_workers=config.num_data_workers,
        data_reader_fn=data_reader_fn,
        model_fn=model_fn,
    )

    # Collect all input files
    input_files = []
    for pattern in config.glob_patterns:
        input_files.extend(list(config.input_dir.glob(pattern)))

    # Set the parsl compute settings
    parsl_config = config.compute_settings.get_config(
        config.output_dir / 'parsl',
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        pool.map(worker_fn, input_files)
