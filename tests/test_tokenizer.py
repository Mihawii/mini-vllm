def test_roundtrip_is_exact(tokenizer):
    text = "The greenhouse robot inspected the tomatoes."
    ids = tokenizer.encode(text)
    assert ids, "encoding produced no tokens"
    assert tokenizer.decode(ids, skip_special_tokens=False) == text


def test_breakdown_matches_encoding(tokenizer):
    text = "Hello world, KV cache!"
    infos = tokenizer.breakdown(text)
    assert [i.token_id for i in infos] == tokenizer.encode(text)
    assert [i.index for i in infos] == list(range(len(infos)))
    # decoded fragments reassemble the original text
    assert "".join(i.text for i in infos) == text


def test_pad_token_falls_back_to_eos(tokenizer):
    # GPT-2 ships without a pad token; the wrapper must fill it in.
    assert tokenizer.pad_token_id is not None
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.tokenizer.padding_side == "left"


def test_vocab_size_positive(tokenizer):
    assert tokenizer.vocab_size >= 50257


def test_chat_template_fallback(tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    )
    assert "Be brief." in rendered
    assert rendered.rstrip().endswith("Assistant:")
