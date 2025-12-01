from .openai import generate as openai_generate
from .hf import generate as hf_generate

__all__ = ["openai_generate", "hf_generate"]
