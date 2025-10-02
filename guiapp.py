"""
Запускаем приложение с веб-сервером на FastAPI
"""
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware

from GuiBackend.studiesAPI import studiesRouter
from GuiBackend.auth import authRouter
from GuiBackend.API import file_router

from AiriApp import AiriFMApp
import logging





from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_: FastAPI):

    airiapp = AiriFMApp()
    asyncio.create_task(airiapp.Run())

    loger = logging.getLogger("AiriFM")
    loger.info("Запускаем сервер")

    yield {"app": airiapp, "loger": loger}

    airiapp.Release()


middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(lifespan=lifespan, middleware=middleware)
app.include_router(studiesRouter)
app.include_router(file_router)
app.include_router(authRouter)

