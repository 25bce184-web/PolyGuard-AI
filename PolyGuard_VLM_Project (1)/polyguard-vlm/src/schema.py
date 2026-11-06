from pydantic import BaseModel, ConfigDict, Field, model_validator

class Medicine(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    drug_name: str
    dosage: str | None
    frequency: str | None
    duration: str | None
    route: str | None
    instructions: str | None
    confidence: float = Field(ge=0, le=1)
    unclear_text: str

    @model_validator(mode="after")
    def conservative(self):
        if not self.drug_name.strip():
            raise ValueError("Medicine name cannot be empty; use UNCLEAR")
        if self.drug_name.strip().upper() == "UNCLEAR" and self.confidence > 0.3:
            raise ValueError("UNCLEAR medicine cannot have high confidence")
        return self

class Prescription(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    raw_text: str
    medicines: list[Medicine]
    overall_confidence: float = Field(ge=0, le=1)
    contains_unclear_content: bool
    notes: str

    @model_validator(mode="after")
    def uncertainty_flag(self):
        uncertain = "UNCLEAR" in self.raw_text.upper() or any(
            m.unclear_text.strip() or any(isinstance(v, str) and "UNCLEAR" in v.upper()
                                        for v in m.model_dump().values())
            for m in self.medicines)
        if uncertain and not self.contains_unclear_content:
            raise ValueError("Unclear content requires contains_unclear_content=true")
        return self
