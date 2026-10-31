"""Pydantic v2 schemas.

Placeholders for later milestones (add here, keep one file until it gets large):
  - Weather / MandiPrice / SoilProfile  (live-data tools, each with source + fetched_at)
  - AdvisoryRequest / AdvisoryResponse  (text + audio + schedule + sell-or-hold)
  - SafetyAlert                         (produced by guards.py, never by the LLM)
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    environment: str
    time: datetime
