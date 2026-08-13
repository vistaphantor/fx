"""
Vista Kaggle BPE Chat — Interactive console for the 50M BPE Reasoning Model.

Supports:
  - 8K BPE Tokenizer (pure Python, standalone)
  - VistaReasoningGPT (PyTorch 50.3M Parameters)
  - Native CoT (<think> ... </think>) formatting
  - Works on Kaggle GPU/CPU or local PC

Usage:
  python tools/kaggle_chat.py
  python tools/kaggle_chat.py --model /kaggle/working/vista_50m_best.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ── Model Architecture ────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d=512, h=16, ffn=2048, drop=0.1):
        super().__init__()
        self.nh = h; self.hd = d // h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d), nn.Dropout(drop))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, C)))
        return x + self.ff(self.ln2(x))

class VistaReasoningGPT(nn.Module):
    def __init__(self, V=8192, d=512, L=16, H=16, ffn=2048, S=512, drop=0.0):
        super().__init__()
        self.vocab_size = V
        self.max_seq_len = S
        self.tok_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Embedding(S, d)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([TransformerBlock(d, H, ffn, drop) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for b in self.blocks: x = b(x)
        logits = self.lm_head(self.ln_f(x))
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=350, temperature=0.75, top_k=50, top_p=0.92, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cp = torch.cumsum(sp, dim=-1)
                sp[(cp - sp) > top_p] = 0.0
                sp /= sp.sum(dim=-1, keepdim=True)
                nxt = si.gather(-1, torch.multinomial(sp, 1))
            else:
                nxt = torch.multinomial(probs, 1)

            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return idx

# ── Standalone Official BPETokenizer (Matches src/language/tokenizer.py) ──────
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"
THINK = "<think>"
ENDTHINK = "</think>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, SEP, THINK, ENDTHINK]

class BPETokenizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self._merge_map: dict[tuple[str, str], str] = {}
        self._word_cache: dict[str, list[int]] = {}

    def _tokenize_word(self, word: str) -> list[int]:
        if word in self._word_cache:
            return self._word_cache[word]

        syms = list(word) + ["</w>"]
        for pair, merged in self._merge_map.items():
            new_syms = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and (syms[i], syms[i + 1]) == pair:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms

        unk_id = self.vocab.get(UNK, 1)
        token_ids = [self.vocab.get(s, unk_id) for s in syms]
        self._word_cache[word] = token_ids
        return token_ids

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        tokens = []
        if add_bos:
            tokens.append(self.vocab.get(BOS, 2))

        space_id = self.vocab.get(" ", self.vocab.get(UNK, 1))
        words = re.findall(r'\S+|\n', text)
        for w in words:
            tokens.extend(self._tokenize_word(w))
            tokens.append(space_id)

        if add_eos:
            tokens.append(self.vocab.get(EOS, 3))
        return tokens

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, UNK)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        text = "".join(tokens)
        text = text.replace("</w>", " ").strip()
        return text

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tok.id_to_token = {int(v): k for k, v in data["vocab"].items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._merge_map = {tuple(m): m[0] + m[1] for m in data["merges"]}
        return tok

    @property
    def vocab_size(self) -> int: return len(self.vocab)
    def pad_id(self) -> int:     return self.vocab.get(PAD, 0)
    def unk_id(self) -> int:     return self.vocab.get(UNK, 1)
    def bos_id(self) -> int:     return self.vocab.get(BOS, 2)
    def eos_id(self) -> int:     return self.vocab.get(EOS, 3)

# ── Search & Load Checkpoint ──────────────────────────────────────────────────
SEARCH_PATHS = [
    "/kaggle/working/vista_50m_best.pt",
    "data/models/language_50m/vista_50m_best.pt",
    "data/models/language_50m/vista_50m.pt",
    "vista_50m_best.pt",
    "/kaggle/input/models/victormutwirib/vistae/pytorch/default/1/fx/data/models/language_50m/vista_50m_best.pt",
]

TOK_SEARCH_PATHS = [
    "data/models/language_50m/tokenizer.json",
    "data/models/language/tokenizer.json",
    "/kaggle/working/fx/data/models/language_50m/tokenizer.json",
    "/kaggle/input/models/victormutwirib/vistae/pytorch/default/1/fx/data/models/language_50m/tokenizer.json",
]

def load_bpe_chat(model_override: str | None = None):
    # Find model
    ckpt_path = None
    candidates = [model_override] if model_override else SEARCH_PATHS
    for c in candidates:
        if c and Path(c).exists():
            ckpt_path = Path(c)
            break

    if not ckpt_path:
        print("❌ Error: Checkpoint not found. Looked in:")
        for c in candidates:
            if c: print(f"  - {c}")
        return None, None

    # Find tokenizer
    tok = None
    for tp in TOK_SEARCH_PATHS:
        if Path(tp).exists():
            print(f"📦 Tokenizer: {tp}")
            tok = BPETokenizer.load(tp)
            break

    if not tok:
        print("❌ Error: BPE tokenizer.json not found.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Loading Checkpoint ({device}): {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt.get("config", {
        "vocab_size": tok.vocab_size,
        "d_model": 512, "n_layers": 16, "n_heads": 16,
        "ffn_dim": 2048, "max_seq_len": 512, "dropout": 0.0
    })

    model = VistaReasoningGPT(
        V=tok.vocab_size,
        d=cfg.get("d_model", 512),
        L=cfg.get("n_layers", 16),
        H=cfg.get("n_heads", 16),
        ffn=cfg.get("ffn_dim", 2048),
        S=cfg.get("max_seq_len", 512),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Loaded: {n_params/1e6:.1f}M Parameters | Epoch: {ckpt.get('epoch', '?')} | Best Loss: {ckpt.get('loss', '?'):.4f}")
    return model, tok

# ── Interactive Chat Loop ─────────────────────────────────────────────────────
BANNER = """
╔═══════════════════════════════════════════════════════════╗
║     VISTA-50M REASONING AI  —  BPE Interactive Chat       ║
║     50.3M Parameters  |  Subword BPE Tokenizer (8K)       ║
╚═══════════════════════════════════════════════════════════╝
  Type your question and press Enter.
  Commands: /quit  /reset  /temp <0.1-2.0>  /top_k <n>
"""

def chat_loop(model: VistaReasoningGPT, tok: FastBPE):
    print(BANNER)
    history = []
    device = next(model.parameters()).device
    temp, top_k, top_p = 0.75, 50, 0.92

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input: continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0]
            if cmd == "/quit": print("Goodbye!"); break
            elif cmd == "/reset": history = []; print("[Conversation history cleared]")
            elif cmd == "/temp" and len(parts) > 1: temp = float(parts[1]); print(f"[Temperature = {temp}]")
            elif cmd == "/top_k" and len(parts) > 1: top_k = int(parts[1]); print(f"[Top_K = {top_k}]")
            else: print("Commands: /quit  /reset  /temp <val>  /top_k <val>")
            continue

        history.append(f"Human: {user_input}")
        if len(history) > 6: history = history[-6:]

        prompt = "\n\n".join(history) + "\n\nAssistant: <think>\n"
        prompt_ids = tok.encode(prompt, add_bos=True)
        inp_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        print("\nVista: <think>", end="\n", flush=True)

        out_tensor = model.generate(
            inp_tensor,
            max_new_tokens=350,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            eos_id=tok.eos_id,
        )

        full_output = tok.decode(out_tensor[0].tolist())
        
        # Extract Assistant portion
        if "Assistant:" in full_output:
            resp = full_output.split("Assistant:")[-1].strip()
        else:
            resp = full_output.strip()

        if "Human:" in resp:
            resp = resp[:resp.index("Human:")].strip()

        print(resp[7:] if resp.startswith("<think>") else resp)
        history.append(f"Assistant: {resp}")


def generate_single_prompt(model: VistaReasoningGPT, tok: FastBPE, prompt_text: str, temp=0.75, top_k=50, top_p=0.92):
    device = next(model.parameters()).device
    prompt = f"Human: {prompt_text}\n\nAssistant: <think>\n"
    prompt_ids = tok.encode(prompt, add_bos=True)
    inp_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\nUser: {prompt_text}")
    print("Vista: <think>", flush=True)

    with torch.no_grad():
        out_tensor = model.generate(
            inp_tensor,
            max_new_tokens=350,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            eos_id=tok.eos_id,
        )

    full_output = tok.decode(out_tensor[0].tolist())
    if "Assistant:" in full_output:
        resp = full_output.split("Assistant:")[-1].strip()
    else:
        resp = full_output.strip()

    if "Human:" in resp:
        resp = resp[:resp.index("Human:")].strip()

    print(resp[7:] if resp.startswith("<think>") else resp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to checkpoint")
    parser.add_argument("--prompt", "-p", default=None, help="Single question to ask (great for Kaggle cells)")
    parser.add_argument("--temp", type=float, default=0.75, help="Sampling temperature")
    args = parser.parse_args()

    model, tok = load_bpe_chat(args.model)
    if model and tok:
        if args.prompt:
            generate_single_prompt(model, tok, args.prompt, temp=args.temp)
        else:
            chat_loop(model, tok)

if __name__ == "__main__":
    main()
