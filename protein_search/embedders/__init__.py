"""Embedder module for embedding sequences."""

from __future__ import annotations

from typing import Any

from protein_search.embedders.auto import AutoEmbedder
from protein_search.embedders.auto import AutoEmbedderConfig
from protein_search.embedders.base import BaseEmbedder
from protein_search.embedders.base import BaseEmbedderConfig
from protein_search.embedders.esm2 import Esm2Embedder
from protein_search.embedders.esm2 import Esm2EmbedderConfig
from protein_search.registry import registry

EmbedderConfigTypes = Esm2EmbedderConfig | AutoEmbedderConfig
EmbedderTypes = Esm2Embedder | AutoEmbedder

_EmbedderTypes = tuple[type[EmbedderConfigTypes], type[EmbedderTypes]]

STRATEGIES: dict[str, _EmbedderTypes] = {
    'esm2': (Esm2EmbedderConfig, Esm2Embedder),
    'auto': (AutoEmbedderConfig, AutoEmbedder),
}


# Make a function to combine the config and embedder initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: dict[str, Any]) -> EmbedderTypes:
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)  # type: ignore[arg-type]
    if not strategy:
        raise ValueError(f'Unknown embedder name: {name}')

    # Unpack the embedder strategy
    config_cls, embedder_cls = strategy

    # Create the embedder config
    config = config_cls(**kwargs)
    # Create the embedder instance
    return embedder_cls(config)


def get_embedder(
    embedder_kwargs: dict[str, Any],
    register: bool = False,
) -> EmbedderTypes:
    """Get the embedder instance based on the embedder name and kwargs.

    Caches the embedder instance based on the embedder name and kwargs.
    Currently supports the following embedders: esm2, auto.

    Parameters
    ----------
    embedder_kwargs : dict[str, Any]
        The embedder configuration. Contains an extra `name` argument
        to specify the embedder to use.
    register : bool, optional
        Register the embedder instance for warmstart, by default False.

    Returns
    -------
    EmbedderTypes
        The embedder instance.

    Raises
    ------
    ValueError
        If the embedder name is unknown.
    """
    # Register and create the embedder instance
    if register:
        registry.register(_factory_fn)
        embedder = registry.get(_factory_fn, **embedder_kwargs)
    else:
        embedder = _factory_fn(**embedder_kwargs)

    return embedder
