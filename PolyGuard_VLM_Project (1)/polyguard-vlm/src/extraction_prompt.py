from .config import ROOT

def get_prompt() -> str:
    return (ROOT / "prompts" / "extract.txt").read_text(encoding="utf-8")
