"""HRM-backed autoregressive text generator.

Architecture: byte -> learned field embedding -> HRM core dynamics ->
pooled field -> vocab logits -> next byte.

The HRM core (field diffusion/reaction, memory, hierarchy, cognition)
provides the recurrent dynamics; learnable interfaces connect it to text.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from hrm.baseline import BASELINE_CONFIG, FULL, State, hrm_step, make_params


class HRMTextGenerator:
    def __init__(self, vocab_size: int = 257, seed: int = 0):
        self.cfg = BASELINE_CONFIG
        self.shape = self.cfg.shape
        self.dyn = self.cfg.dynamics()
        self.vocab_size = vocab_size
        self.eos_id = 256
        self.batch = 1

        key = random.PRNGKey(seed)
        k1, k2, k3 = random.split(key, 3)
        # HRM core params (frozen after Stage 1 training, or fine-tuned)
        self.hrm_params = make_params(k1, self.shape)
        # Learnable byte -> field embedding: (vocab, h, w, c)
        s = self.shape
        self.byte_embed = random.normal(k2, (vocab_size, s.height, s.width, s.channels), dtype=jnp.float32) * 0.1
        # Learnable head: pooled field (c) -> vocab
        self.vocab_head = random.normal(k3, (s.channels, vocab_size), dtype=jnp.float32) * 0.1

    def _make_state(self, key):
        s = self.shape
        k1, k2 = random.split(key)
        phi = 0.05 * random.normal(k1, (self.batch, s.height, s.width, s.channels), dtype=jnp.float32)
        memory = jnp.zeros_like(phi)
        cognition = 0.01 * random.normal(k2, (self.batch, s.cognitive_dim), dtype=jnp.float32)
        hierarchy = jnp.zeros((self.batch, s.hierarchy_height, s.hierarchy_width, s.channels), dtype=jnp.float32)
        allocation = jnp.ones((self.batch, s.height, s.width, s.channels), dtype=jnp.float32)
        return State(phi, memory, cognition, hierarchy, allocation)

    def _forward(self, hrm_params, byte_embed, vocab_head, ids, state):
        """ids: (seq,) -> logits: (seq, vocab)"""
        def body(s, byte_id):
            frame = byte_embed[byte_id][None, :, :, :]  # (1, h, w, c)
            dummy = jnp.zeros_like(frame)
            s2, _ = hrm_step(s, (frame, dummy, dummy), hrm_params, self.dyn, self.shape, FULL)
            pooled = s2.phi.mean(axis=(1, 2))[0]  # (c,)
            logits = pooled @ vocab_head  # (vocab,)
            return s2, logits
        _, logits = jax.lax.scan(body, state, ids)
        return logits

    def train(self, text: str, epochs: int = 100, seq_len: int = 64,
              lr: float = 0.03, seed: int = 0, train_hrm: bool = False):
        data = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).astype(np.int32)
        if len(data) < seq_len + 1:
            raise ValueError(f"Need {seq_len+1} bytes, got {len(data)}")
        key = random.PRNGKey(seed)
        history = []
        for epoch in range(epochs):
            key, skey = random.split(key)
            start = int(random.randint(skey, (), 0, len(data) - seq_len))
            window = jnp.asarray(data[start:start + seq_len + 1])
            inp, tgt = window[:-1], window[1:]
            key, skey2 = random.split(key)
            state = self._make_state(skey2)
            if train_hrm:
                def loss_fn(hp, be, vh):
                    logits = self._forward(hp, be, vh, inp, state)
                    logp = jax.nn.log_softmax(logits, axis=-1)
                    return jnp.mean(-jnp.take_along_axis(logp, tgt[:, None], axis=-1).squeeze(-1))
                lv, (gh, gb, gv) = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(
                    self.hrm_params, self.byte_embed, self.vocab_head)
                self.hrm_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, self.hrm_params, gh)
                self.byte_embed = self.byte_embed - lr * gb
                self.vocab_head = self.vocab_head - lr * gv
            else:
                def loss_fn(be, vh):
                    logits = self._forward(self.hrm_params, be, vh, inp, state)
                    logp = jax.nn.log_softmax(logits, axis=-1)
                    return jnp.mean(-jnp.take_along_axis(logp, tgt[:, None], axis=-1).squeeze(-1))
                lv, (gb, gv) = jax.value_and_grad(loss_fn, argnums=(0, 1))(self.byte_embed, self.vocab_head)
                self.byte_embed = self.byte_embed - lr * gb
                self.vocab_head = self.vocab_head - lr * gv
            history.append(float(lv))
        return history

    def generate(self, prompt: str = "", max_tokens: int = 200, temperature: float = 0.8, seed: int = 0) -> str:
        key = random.PRNGKey(seed)
        state = self._make_state(key)
        pbytes = prompt.encode("utf-8", errors="replace")
        ids = list(pbytes)
        # Prime
        for bid in ids:
            frame = self.byte_embed[int(bid)][None, :, :, :]
            dummy = jnp.zeros_like(frame)
            state, _ = hrm_step(state, (frame, dummy, dummy), self.hrm_params, self.dyn, self.shape, FULL)
        # Generate
        for _ in range(max_tokens):
            prev = ids[-1] if ids else 0
            frame = self.byte_embed[int(prev)][None, :, :, :]
            dummy = jnp.zeros_like(frame)
            state, _ = hrm_step(state, (frame, dummy, dummy), self.hrm_params, self.dyn, self.shape, FULL)
            pooled = state.phi.mean(axis=(1, 2))[0]
            logits = (pooled @ self.vocab_head) / max(temperature, 1e-6)
            key, skey = random.split(key)
            nid = int(random.categorical(skey, logits))
            if nid == self.eos_id:
                break
            ids.append(nid)
        gen = ids[len(pbytes):] if prompt else ids
        return bytes([i for i in gen if 0 <= i < 256]).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        leaves = jax.tree_util.tree_leaves(self.hrm_params)
        np.savez(path / "hrm_textgen.npz",
                 **{f"h_{i}": np.asarray(l) for i, l in enumerate(leaves)},
                 byte_embed=np.asarray(self.byte_embed),
                 vocab_head=np.asarray(self.vocab_head))

    def load(self, path: str | Path) -> None:
        d = np.load(Path(path) / "hrm_textgen.npz")
        leaves, treedef = jax.tree_util.tree_flatten(self.hrm_params)
        new_leaves = [jnp.asarray(d[f"h_{i}"]) for i in range(len(leaves))]
        self.hrm_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
        self.byte_embed = jnp.asarray(d["byte_embed"])
        self.vocab_head = jnp.asarray(d["vocab_head"])
