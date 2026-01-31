"""Registry pattern for blocks and attention mechanisms."""

from typing import Dict, Type, Callable, Any
import torch.nn as nn


class BlockRegistry:
    """Registry for decoder block types."""

    _blocks: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        """Register a block class with a name."""
        def decorator(block_class: Type[nn.Module]) -> Type[nn.Module]:
            cls._blocks[name] = block_class
            return block_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """Get a block class by name."""
        if name not in cls._blocks:
            available = list(cls._blocks.keys())
            raise ValueError(f"Unknown block type: '{name}'. Available: {available}")
        return cls._blocks[name]

    @classmethod
    def available(cls) -> list:
        """List available block types."""
        return list(cls._blocks.keys())


class AttentionRegistry:
    """Registry for cross-attention types."""

    _attentions: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        """Register an attention class with a name."""
        def decorator(attn_class: Type[nn.Module]) -> Type[nn.Module]:
            cls._attentions[name] = attn_class
            return attn_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """Get an attention class by name."""
        if name not in cls._attentions:
            available = list(cls._attentions.keys())
            raise ValueError(f"Unknown attention type: '{name}'. Available: {available}")
        return cls._attentions[name]

    @classmethod
    def available(cls) -> list:
        """List available attention types."""
        return list(cls._attentions.keys())
