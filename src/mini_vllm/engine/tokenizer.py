"""Tokenizer wrapper.

We use Hugging Face tokenizers for the vocabulary and BPE merges, but keep a
thin wrapper so the rest of the engine never touches tokenizer internals.

Two decisions matter for batching:
- GPT-2 family models ship without a pad token, so we reuse EOS as pad.
- Padding side is LEFT. In a decoder-only model the next token is predicted
  from the LAST position, so all real tokens must be flush against the right
  edge of the batch tensor. Left padding plus an attention mask achieves that.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass
class TokenInfo:
    """One row of the tokenizer inspector table."""

    index: int
    token_id: int
    token: str  # raw BPE piece, e.g. "Ġgreenhouse" (Ġ marks a leading space)
    text: str  # decoded text fragment, e.g. " greenhouse"


class TokenizerWrapper:
    def __init__(self, name_or_path: str):
        self.name = name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    # -- core operations ---------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def breakdown(self, text: str) -> list[TokenInfo]:
        """Per-token view used by `mini-vllm tokenize` and the dashboard."""
        ids = self.encode(text)
        pieces = self.tokenizer.convert_ids_to_tokens(ids)
        return [
            TokenInfo(index=i, token_id=tid, token=piece, text=self.tokenizer.decode([tid]))
            for i, (tid, piece) in enumerate(zip(ids, pieces))
        ]

    # -- metadata ----------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def apply_chat_template(self, messages: list[dict]) -> str:
        """Render chat messages to a single prompt string.

        Models with a real chat template (e.g. TinyLlama) use it. GPT-2 family
        models have none, so we fall back to a plain readable transcript and
        document that in docs/api.md.
        """
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        lines.append("Assistant:")
        return "\n".join(lines)
