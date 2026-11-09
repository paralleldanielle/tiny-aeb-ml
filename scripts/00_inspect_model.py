"""M1: inspect SmolLM2-135M by following one prompt through the model, bottom to top.

Each section matches one step up the diagram in ``docs/MODEL.md``. Run with
``uv run python scripts/00_inspect_model.py``.
"""

import inspect
from pathlib import Path

import torch
from torch.nn import functional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from tiny_aeb_ml import runtime

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "Ego speed 72 km/h. Car ahead at 38 m, driving 40 km/h. Decision?"
COMPLETION = "v_ego=20.0 m/s\nv_lead=11.1 m/s\nv_rel=8.9 m/s\nTTC=38.0/8.9=4.3 s\nDECISION: WARN"
SNIPPETS = [
    "72",
    " 72",
    "38.0",
    "4.3",
    "km/h",
    " m/s",
    "TTC",
    " WARN",
    " BRAKE",
    " EMERGENCY",
    # Rare characters fall back to their UTF-8 bytes: one character can become several tokens.
    "über",
    "°",
    "😀",
]
# Single-token words whose nearest neighbours in the embedding table we look at in step 2.
NEIGHBOUR_WORDS = ["7", " car", " km", " brake"]
NUM_NEIGHBOURS = 6
# Prompt position of '7', the token step 3 follows through block 1.
SEVEN_POSITION = 4
SHOWN_VALUES = 4
# Blocks (1-based) whose input size step 3 reports; 31 stands for the final norm.
REPORTED_BLOCKS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 31]
# Prompt positions shown in step 3a: 'E', '␣speed', '7', '␣km', '␣Decision'.
RMSNORM_POSITIONS = [0, 2, 4, 6, 26]
# Step 3b-1 prints the attention table for the first few tokens only.
ATTENTION_TABLE_SIZE = 8
# Second digits of 72, 38 and 40: what does the previous-token head make them look at?
PREVIOUS_TOKEN_EXAMPLES = [5, 15, 21]


def show_pieces(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> str:
    """Render token ids as |piece|piece|, with spaces as ␣ and newlines as ↵ so they are visible."""
    pieces = [tokenizer.decode([token_id]) for token_id in ids]
    return "|" + "|".join(p.replace(" ", "␣").replace("\n", "↵") for p in pieces) + "|"


def rms(x: torch.Tensor) -> torch.Tensor:
    """Root-mean-square over the last dimension: the typical size of one entry in each vector."""
    return x.pow(2).mean(dim=-1).sqrt()


def first_values(vector: torch.Tensor) -> str:
    """Format the first few entries of a vector, e.g. ``[-0.023 -0.062 ...]``."""
    return "[" + " ".join(f"{value:+.3f}" for value in vector[:SHOWN_VALUES].tolist()) + " ...]"


def source_location(cls: type) -> str:
    """Return ``path:line`` where a class is defined, relative to site-packages."""
    path = Path(inspect.getsourcefile(cls) or "?")
    line = inspect.getsourcelines(cls)[1]
    return f"{path.as_posix().split('site-packages/')[-1]}:{line}"


def section_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
    """Step 1: show how the prompt and the target completion split into tokens."""
    print("=== Step 1: tokens ===")
    print(f"vocabulary size: {tokenizer.vocab_size:,}")
    for name, text in [("prompt", PROMPT), ("completion", COMPLETION)]:
        ids = tokenizer(text)["input_ids"]
        print(f"\n{name}: {len(text)} characters -> {len(ids)} tokens")
        print(show_pieces(tokenizer, ids))
        print(f"ids: {ids}")

    print("\nsnippets:")
    for text in SNIPPETS:
        ids = tokenizer(text)["input_ids"]
        print(f"  {text!r:14} -> {len(ids)} tokens {show_pieces(tokenizer, ids):28} ids {ids}")

    # EOS ("end of sequence") is the token that tells generate() to stop. The model only learns
    # to emit it if every training target ends with it.
    print(f"\nEOS token: {tokenizer.eos_token!r}, id {tokenizer.eos_token_id}")


def section_embedding(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> None:
    """Step 2: show the embedding table, the lookup for our prompt, and which tokens sit close."""
    print("\n=== Step 2: embedding ===")
    embedding = model.get_input_embeddings()
    table = embedding.weight  # [vocab, hidden] = [49152, 576], float32
    print(f"embedding table: {tuple(table.shape)} = {table.numel():,} numbers")
    size = rms(table.flatten())
    print(f"values range {table.min():+.2f} .. {table.max():+.2f}, typical size (RMS) {size:.3f}")

    seven_id = tokenizer.convert_tokens_to_ids("7")
    first_values = " ".join(f"{value:+.3f}" for value in table[seven_id, :8].tolist())
    print(f"token '7' is id {seven_id}, so its vector is row {seven_id} of the table:")
    print(f"  [{first_values} ...] (576 numbers)")

    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]  # [1, 28], int64
    vectors = embedding(ids)  # [1, 28, 576], float32: one row looked up per token
    print(f"\nprompt ids {tuple(ids.shape)} -> vectors {tuple(vectors.shape)}")

    # The lookup knows nothing about position: the same token gives the same vector wherever it
    # appears. Position only enters later, through RoPE inside attention.
    km_id = tokenizer.convert_tokens_to_ids("Ġkm")
    positions = (ids[0] == km_id).nonzero().flatten().tolist()
    same = torch.equal(vectors[0, positions[0]], vectors[0, positions[1]])
    print(f"'␣km' appears at positions {positions}; identical vectors: {same}")

    # Cosine similarity compares directions and ignores length: 1 = same direction, 0 = unrelated.
    unit_rows = functional.normalize(table, dim=1)
    print(f"\nnearest neighbours by cosine similarity (top {NUM_NEIGHBOURS - 1}):")
    for word in NEIGHBOUR_WORDS:
        word_ids = tokenizer(word)["input_ids"]
        if len(word_ids) != 1:
            raise ValueError(f"{word!r} is {len(word_ids)} tokens; pick a single-token word")
        similarities = unit_rows @ unit_rows[word_ids[0]]  # [vocab]
        top = similarities.topk(NUM_NEIGHBOURS)
        # The first hit is always the word itself (similarity 1.0), so skip it.
        neighbours = [
            f"{show_pieces(tokenizer, [token_id])} {score:.2f}"
            for score, token_id in zip(
                top.values[1:].tolist(), top.indices[1:].tolist(), strict=True
            )
        ]
        print(f"  {show_pieces(tokenizer, word_ids):10} -> {'  '.join(neighbours)}")


def section_blocks(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> None:
    """Step 3: show that each block updates every token's vector by adding to it."""
    print("\n=== Step 3: blocks and the residual stream ===")
    blocks = model.model.layers
    print(f"{len(blocks)} blocks ({type(blocks[0]).__name__}), each with its own weights,")
    print("run one after another: block 1's output is block 2's input")
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]  # [1, 28]

    # Hooks are callbacks PyTorch runs around a layer. The forward hooks record the two updates
    # block 1 computes; the pre-hooks record each norm's input, i.e. the residual stream as it
    # enters that block.
    updates: dict[str, torch.Tensor] = {}
    stream: list[torch.Tensor] = []
    norms = [block.input_layernorm for block in blocks] + [model.model.norm]
    hooks = [
        blocks[0].self_attn.register_forward_hook(
            lambda module, args, output: updates.update(attention=output[0][0])
        ),
        blocks[0].mlp.register_forward_hook(
            lambda module, args, output: updates.update(feed_forward=output[0])
        ),
    ] + [
        layer_norm.register_forward_pre_hook(lambda module, args: stream.append(args[0][0]))
        for layer_norm in norms
    ]
    outputs = model(ids, output_hidden_states=True)
    for hook in hooks:
        hook.remove()

    # hidden_states[0] is the embedding output, hidden_states[1] the output of block 1.
    vector_in = outputs.hidden_states[0][0]  # [28, 576]
    after_attention = vector_in + updates["attention"]
    after_feed_forward = after_attention + updates["feed_forward"]
    piece = show_pieces(tokenizer, [int(ids[0, SEVEN_POSITION])])
    print(f"\ntoken {piece} (position {SEVEN_POSITION}) through block 1, first numbers only:")
    rows = [
        ("vector in (its embedding)", vector_in),
        ("+ attention update", updates["attention"]),
        ("= after add 1", after_attention),
        ("+ feed-forward update", updates["feed_forward"]),
        ("= after add 2 = block 1 out", after_feed_forward),
    ]
    for label, tensor in rows:
        print(f"  {label:28} {first_values(tensor[SEVEN_POSITION])}  shape {tuple(tensor.shape)}")
    difference = (after_feed_forward - outputs.hidden_states[1][0]).abs().max().item()
    print(f"rebuilt by hand vs. the model's block 1 output: max difference {difference:.1e}")

    print("\nsize (RMS, median over the 28 tokens) of the residual stream entering:")
    for block in REPORTED_BLOCKS:
        label = "final norm" if block == len(norms) else f"block {block}"
        print(f"  {label:11} {rms(stream[block - 1]).median():6.2f}")
    first_token = torch.stack([rms(entry[0]) for entry in stream])  # [31]
    peak_block = int(first_token.argmax()) + 1
    peak = first_token.max()
    print(f"(token 0 'E' is an outlier: it peaks at {peak:.0f} entering block {peak_block})")


def section_rmsnorm(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> None:
    """Step 3a: show what RMSNorm does to the vectors entering block 1, and why it is needed."""
    print("\n=== Step 3a: RMSNorm ===")
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]  # [1, 28]
    x = model.get_input_embeddings()(ids)[0]  # [28, 576]: the input to block 1
    norm = model.model.layers[0].input_layernorm

    # The same two steps as LlamaRMSNorm.forward, by hand: divide each vector by its RMS, then
    # multiply each of the 576 dimensions by its own learned gain.
    normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm.variance_epsilon)
    output = normalized * norm.weight

    print("RMSNorm 1 of block 1, applied to each token's vector on its own:")
    print(f"  {'token':12} {'RMS in':>8} {'÷ RMS':>8} {'× gain':>8}")
    for position in RMSNORM_POSITIONS:
        piece = show_pieces(tokenizer, [int(ids[0, position])])
        print(
            f"  {piece:12} {rms(x[position]):8.3f} {rms(normalized[position]):8.3f}"
            f" {rms(output[position]):8.3f}"
        )
    gain = norm.weight  # [576]
    print(f"learned gain: 576 numbers, range {gain.min():+.2f} .. {gain.max():+.2f}")
    difference = (output - norm(x)).abs().max().item()
    print(f"by hand vs. the model's own layer: max difference {difference:.1e}")

    # Gains belong to the norm layer, not to a token: every token passing through a given norm is
    # scaled the same way, and each of the 61 norms in the model has its own 576 gains.
    blocks = model.model.layers
    print(f"\nfirst {SHOWN_VALUES} gains of three different norms (shared by all tokens):")
    for label, layer_norm in [
        ("block 1, RMSNorm 1", blocks[0].input_layernorm),
        ("block 1, RMSNorm 2", blocks[0].post_attention_layernorm),
        ("block 2, RMSNorm 1", blocks[1].input_layernorm),
    ]:
        print(f"  {label:20} {first_values(layer_norm.weight)}")
    print(f"\nsource: {source_location(type(norm))} (LlamaRMSNorm)")
    print(inspect.getsource(type(norm).forward))


def section_attention(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> None:
    """Step 3b-1: show real attention weights: the causal mask, one readable head, token 0."""
    print("\n=== Step 3b-1: attention weights ===")
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]  # [1, 28]
    attentions = model(ids, output_attentions=True).attentions  # 30 x [1, heads, 28, 28]
    num_heads = attentions[0].shape[1]
    print(f"{len(attentions)} blocks x {num_heads} heads, each a 28 x 28 table of weights")
    pieces = [show_pieces(tokenizer, [token_id]).strip("|") for token_id in ids[0].tolist()]

    size = ATTENTION_TABLE_SIZE
    weights = attentions[0][0, 0]  # block 1, head 1: [query token, key token]
    print(
        f"\nblock 1, head 1, first {size} tokens (row = token looking, column = token looked at):"
    )
    print(" " * 9 + "".join(f"{piece:>7}" for piece in pieces[:size]))
    for row in range(size):
        cells = "".join(
            f"{weights[row, col]:7.2f}" if col <= row else f"{'·':>7}" for col in range(size)
        )
        print(f"  {pieces[row]:>7}{cells}")
    print("every row sums to 1; above the diagonal is 0 because no token may look ahead")

    # The head that most consistently looks at the token right before. Position 1 is skipped:
    # its previous token is token 0, which most heads look at anyway (see below).
    previous_share = torch.stack(
        [
            torch.diagonal(block[0], offset=-1, dim1=-2, dim2=-1)[:, 1:].mean(dim=-1)
            for block in attentions
        ]
    )  # [30 blocks, heads]
    block_index, head_index = divmod(int(previous_share.argmax()), num_heads)
    print(
        f"\nblock {block_index + 1}, head {head_index + 1} puts {previous_share.max():.0%}"
        " of its weight on the previous token:"
    )
    weights = attentions[block_index][0, head_index]
    for position in PREVIOUS_TOKEN_EXAMPLES:
        target = int(weights[position].argmax())
        print(
            f"  {pieces[position]} at {position:>2} looks at {pieces[target]} at {target:>2}"
            f" with weight {weights[position, target]:.2f}"
        )

    # Mean over heads and over tokens 1..27 (token 0 can only look at itself).
    on_first_token = [block[0, :, 1:, 0].mean().item() for block in attentions]
    print("\nshare of all attention that lands on token 0 'E', per block:")
    print("  blocks  1-15: " + " ".join(f"{share:.2f}" for share in on_first_token[:15]))
    print("  blocks 16-30: " + " ".join(f"{share:.2f}" for share in on_first_token[15:]))


def main() -> None:
    """Run the inspection sections in order, bottom of the diagram to the top."""
    runtime.configure_logging()
    runtime.configure_threads()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # "eager" is the plain attention implementation; unlike the default it can return its
    # attention weights, which step 3b shows. The results are the same up to rounding.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    )
    # We only look, never train, so skip gradient bookkeeping.
    with torch.inference_mode():
        section_tokens(tokenizer)
        section_embedding(tokenizer, model)
        section_blocks(tokenizer, model)
        section_rmsnorm(tokenizer, model)
        section_attention(tokenizer, model)


if __name__ == "__main__":
    main()
