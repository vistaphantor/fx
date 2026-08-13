"""
Vista Chat — interactive console for talking to the trained PyTorch 50M Reasoning Model or NumPy model.

Usage:
    python tools/vista_chat.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.tokenizer import BPETokenizer
from src.language.pytorch_transformer import VistaReasoningGPT

MODEL_SEARCH_PATHS = [
    "data/models/language_50m/vista_50m_best.pt",
    "data/models/language_50m/vista_50m.pt",
    "vista_50m_best.pt",
    "/kaggle/working/vista_50m_best.pt",
]
TOK_50M_PATH = "data/models/language_50m/tokenizer.json"

GEN_CONFIG = {
    "max_new_tokens": 350,
    "temperature":    0.75,
    "top_k":          50,
    "top_p":          0.92,
}

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║    VISTA-50M-REASONING  —  Conversational AI (PyTorch)    ║
║    50.3 Million Parameters  |  Multi-Threaded CPU Engine  ║
╚═══════════════════════════════════════════════════════════╝
  Type your prompt and press Enter.
  Commands: /quit  /reset  /temp <0-2>  /top_k <n>  /top_p <p>
"""


def load_50m_model(model_path_override: str | None = None):
    # ── Find checkpoint ────────────────────────────────────────────────────────
    model_path = None
    search_list = [model_path_override] if model_path_override else MODEL_SEARCH_PATHS
    for p in search_list:
        if p and Path(p).exists():
            model_path = Path(p)
            break

    if model_path is None:
        print(f"ERROR: No checkpoint found. Looked in:")
        for p in search_list:
            if p: print(f"  - {p}")
        print("\nIf on PC: Download vista_50m_best.pt from Kaggle Output panel into data/models/language_50m/")
        return None, None

    device = torch.device("cpu")
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # ── Auto-detect model type ─────────────────────────────────────────────────
    if "vocab" in checkpoint and isinstance(checkpoint["vocab"], dict) and len(checkpoint["vocab"]) < 500:
        # This is a Kaggle char-level checkpoint — redirect to kaggle_chat.py
        vocab = checkpoint["vocab"]
        print(f"\n[Detected] Char-level Kaggle model (vocab={len(vocab)} chars, epoch={checkpoint.get('epoch')}, loss={checkpoint.get('loss', 0):.4f})")
        print("[Redirecting] Launching kaggle_chat.py instead...\n")
        import subprocess, sys
        subprocess.run([sys.executable, str(Path(__file__).parent / "kaggle_chat.py"), "--model", str(model_path)])
        return None, None  # kaggle_chat handles everything

    # ── BPE model path ─────────────────────────────────────────────────────────
    print("Loading 8K BPE Tokenizer...")
    tok_path = Path(TOK_50M_PATH)
    if not tok_path.exists():
        tok_path = Path("data/models/language/tokenizer.json")
        if not tok_path.exists():
            print("ERROR: No tokenizer found. Run tools/train_pytorch_50m.py first.")
            return None, None

    tok = BPETokenizer.load(tok_path)
    print(f"  Vocab Size: {tok.vocab_size:,} tokens")

    cfg = checkpoint.get("config", {
        "vocab_size": tok.vocab_size,
        "d_model": 512, "n_layers": 16, "n_heads": 16,
        "ffn_dim": 2048, "max_seq_len": 512, "dropout": 0.0
    })

    print("Loading PyTorch 50M Reasoning Model (BPE)...")
    model = VistaReasoningGPT(
        vocab_size  = tok.vocab_size,
        d_model     = cfg.get("d_model", 512),
        n_layers    = cfg.get("n_layers", 16),
        n_heads     = cfg.get("n_heads", 16),
        ffn_dim     = cfg.get("ffn_dim", 2048),
        max_seq_len = cfg.get("max_seq_len", 512),
        dropout     = 0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Parameters: {model.get_num_params()/1e6:.2f}M  |  Checkpoint: {model_path.name}")
    return model, tok


def chat_loop(model: VistaReasoningGPT, tok: BPETokenizer):
    print(BANNER)
    gen_cfg = dict(GEN_CONFIG)
    history: list[str] = []
    MAX_TURNS = 3

    device = torch.device("cpu")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split()
            if cmd[0] == "/quit":
                print("Goodbye!")
                break
            elif cmd[0] == "/reset":
                history = []
                print("[Conversation history reset]")
            elif cmd[0] == "/temp" and len(cmd) > 1:
                gen_cfg["temperature"] = float(cmd[1])
                print(f"[Temperature set to {gen_cfg['temperature']}]")
            elif cmd[0] == "/top_k" and len(cmd) > 1:
                gen_cfg["top_k"] = int(cmd[1])
                print(f"[top_k set to {gen_cfg['top_k']}]")
            elif cmd[0] == "/top_p" and len(cmd) > 1:
                gen_cfg["top_p"] = float(cmd[1])
                print(f"[top_p set to {gen_cfg['top_p']}]")
            else:
                print("Commands: /quit /reset /temp /top_k /top_p")
            continue

        history.append(f"Human: {user_input}")
        if len(history) > MAX_TURNS * 2:
            history = history[-(MAX_TURNS * 2):]

        prompt = "\n\n".join(history) + "\n\nAssistant: <think>"
        prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        print("\nVista: <think>", end="", flush=True)

        out_tensor = model.generate(
            input_tensor,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_k=gen_cfg["top_k"],
            top_p=gen_cfg["top_p"],
            eos_id=tok.eos_id(),
        )

        full_generated = tok.decode(out_tensor[0].tolist())
        # Extract assistant response after last Assistant: prompt
        if "Assistant:" in full_generated:
            resp = full_generated.split("Assistant:")[-1].strip()
        else:
            resp = full_generated.strip()

        if "Human:" in resp:
            resp = resp[:resp.index("Human:")].strip()

        print(resp[7:] if resp.startswith("<think>") else resp)
        history.append(f"Assistant: {resp}")


def main():
    model, tok = load_50m_model()
    if model is None:
        return
    chat_loop(model, tok)


if __name__ == "__main__":
    main()
