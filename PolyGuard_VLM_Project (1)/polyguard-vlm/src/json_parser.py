"""Remove an enclosing JSON fence, but never repair or guess clinical content."""
import json
import re
from .schema import Prescription

class OutputError(ValueError):
    def __init__(self, message, json_valid=False):
        super().__init__(message)
        self.json_valid = json_valid

def parse_response(raw: str) -> tuple[Prescription, bool]:
    if not raw or not raw.strip():
        raise OutputError("empty_response")
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    try:
        obj = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError):
        raise OutputError("invalid_json") from None
    try:
        return Prescription.model_validate(obj), fenced is not None
    except ValueError:
        # Do not log Pydantic errors: they can include prescription text.
        raise OutputError("invalid_schema", json_valid=True) from None
