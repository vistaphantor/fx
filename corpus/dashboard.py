from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corpus.auditor import CorpusStats


class CorpusDashboard:
    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds <= 0 or seconds > 86400 * 365:
            return "N/A"
        mins, secs = divmod(int(seconds), 60)
        hrs, mins = divmod(mins, 60)
        days, hrs = divmod(hrs, 24)
        if days > 0:
            return f"{days}d {hrs}h {mins}m"
        if hrs > 0:
            return f"{hrs}h {mins}m {secs}s"
        return f"{mins}m {secs}s"

    @staticmethod
    def _progress_bar(pct: float, width: int = 24) -> str:
        filled = max(0, min(width, int(width * (pct / 100.0))))
        empty = width - filled
        return "█" * filled + "░" * empty

    def render(self, stats: "CorpusStats", local_sources: int = 0, hf_sources: int = 0) -> str:
        pct = stats.progress_pct
        pbar = self._progress_bar(pct)
        eta_str = self._format_time(stats.eta_seconds)
        mem_str = f"{stats.memory_mb:.1f} MB" if stats.memory_mb > 0 else "N/A"

        tot_tok = max(1, stats.total_tokens)
        reason_pct = (stats.reasoning_tokens / tot_tok) * 100
        inst_pct   = (stats.instruction_tokens / tot_tok) * 100
        code_pct   = (stats.code_tokens / tot_tok) * 100
        math_pct   = (stats.math_tokens / tot_tok) * 100
        gen_pct    = (stats.general_tokens / tot_tok) * 100

        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║                  VISTA CORPUS STATUS                     ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Sources         Local: {local_sources:<4} | HuggingFace: {hf_sources:<4}    ║",
            f"║  Documents       {stats.total_documents:<39,} ║",
            f"║  Exact Tokens    {stats.total_tokens:<39,} ║",
            f"║  Target          {stats.target_tokens:<39,} ║",
            f"║  Progress        {pct:7.5f}%   {pbar} ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Token Composition                                       ║",
            f"║    Reasoning <think>   {stats.reasoning_tokens:12,}  ({reason_pct:5.1f}%)               ║",
            f"║    Instruction          {stats.instruction_tokens:12,}  ({inst_pct:5.1f}%)               ║",
            f"║    Code                 {stats.code_tokens:12,}  ({code_pct:5.1f}%)               ║",
            f"║    Math                 {stats.math_tokens:12,}  ({math_pct:5.1f}%)               ║",
            f"║    General              {stats.general_tokens:12,}  ({gen_pct:5.1f}%)               ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Quality & Dedup                                         ║",
            f"║    Duplicates Removed    {stats.duplicate_rate * 100:5.1f}%                            ║",
            f"║    Quality Pass Rate    {stats.quality_pass_rate * 100:5.1f}%                            ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Live Scan / Build Metrics                               ║",
            f"║    Docs/sec           {stats.docs_per_sec:<39,.0f} ║",
            f"║    Tokens/sec         {stats.tokens_per_sec:<39,.0f} ║",
            f"║    ETA                {eta_str:<39} ║",
            f"║    Memory Usage       {mem_str:<39} ║",
            f"║    Shards Written     train: {stats.shards_train:<4} val: {stats.shards_val:<4}           ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def display(self, stats: "CorpusStats", local_sources: int = 0, hf_sources: int = 0) -> None:
        rendered = self.render(stats, local_sources, hf_sources)
        try:
            print("\n" + rendered + "\n", flush=True)
        except UnicodeEncodeError:
            ascii_rendered = (
                rendered.replace("╔", "+").replace("╗", "+")
                .replace("╚", "+").replace("╝", "+")
                .replace("╠", "+").replace("╣", "+")
                .replace("═", "-").replace("║", "|")
                .replace("█", "#").replace("░", "-")
            )
            print("\n" + ascii_rendered + "\n", flush=True)
