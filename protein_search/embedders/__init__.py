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

EMBEDDER_STRATEGIES: dict[str, _EmbedderTypes] = {
    'esm2': (Esm2EmbedderConfig, Esm2Embedder),
    'auto': (AutoEmbedderConfig, AutoEmbedder),
}


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
    name = embedder_kwargs.get('name', '')
    embedder_strategy = EMBEDDER_STRATEGIES.get(name)
    if not embedder_strategy:
        raise ValueError(f'Unknown embedder name: {name}')

    # Unpack the embedder strategy
    config_cls, embedder_cls = embedder_strategy
    # Create the embedder config
    config = config_cls(**embedder_kwargs)

    # Register the embedder
    if register:
        registry.register(embedder_cls)
        embedder = registry.get(embedder_cls, config)
    else:
        embedder = embedder_cls(config)

    return embedder  # type: ignore[return-value]

    # if name == 'esm2':
    #     from protein_search.embedders.esm2 import Esm2Embedder

    #     return Esm2Embedder(Esm2EmbedderConfig(**embedder_kwargs))
    # elif name == 'auto':
    #     from protein_search.embedders.auto import AutoEmbedder

    #     return AutoEmbedder(AutoEmbedderConfig(**embedder_kwargs))
    # else:
    #     raise ValueError(f'Unknown embedder name: {name}')
