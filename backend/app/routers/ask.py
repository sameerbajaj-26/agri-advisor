"""POST /ask: question in, structured advisory JSON out."""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from app.agent.guards import Thresholds
from app.agent.loop import run_agent
from app.agent.schemas import AskRequest, AskResponse
from app.agent.tools import ToolRegistry
from app.deps import (
    SettingsDep,
    get_clock,
    get_geocoder,
    get_llm,
    get_mandi,
    get_retriever_factory,
    get_soil,
    get_weather,
)
from app.llm.base import LLMClient
from app.providers.geocode import GeocodeProvider
from app.providers.mandi import MandiProvider
from app.providers.soil import SoilProvider
from app.providers.weather import WeatherProvider
from app.rag.retriever import Retriever

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    settings: SettingsDep,
    llm: Annotated[LLMClient, Depends(get_llm)],
    geocoder: Annotated[GeocodeProvider, Depends(get_geocoder)],
    weather: Annotated[WeatherProvider, Depends(get_weather)],
    mandi: Annotated[MandiProvider, Depends(get_mandi)],
    soil: Annotated[SoilProvider, Depends(get_soil)],
    retriever: Annotated[Callable[[], Retriever], Depends(get_retriever_factory)],
    now: Annotated[datetime, Depends(get_clock)],
) -> AskResponse:
    thresholds = Thresholds.from_settings(settings)  # 503 before any LLM cost if unset
    tools = ToolRegistry(
        geocoder=geocoder, weather=weather, mandi=mandi, soil=soil,
        retriever=retriever, settings=settings,
    )  # fmt: skip
    return await run_agent(
        body,
        llm=llm,
        tools=tools,
        thresholds=thresholds,
        settings=settings,
        now=now,
    )
