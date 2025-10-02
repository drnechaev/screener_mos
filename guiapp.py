"""
Запускаем приложение с веб-сервером на FastAPI
"""
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from GuiBackend.studiesAPI import studiesRouter
from GuiBackend.auth import authRouter
from GuiBackend.API import file_router

from AiriApp import AiriFMApp
import logging


origins = {
    "*",
    "http://0.0.0.0",
    "http://localhost",
}



from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_: FastAPI):

    airiapp = AiriFMApp()
    asyncio.create_task(airiapp.Run())

    loger = logging.getLogger("AiriFM")
    loger.info("Запускаем сервер")

    yield {"app": airiapp, "loger": loger}

    airiapp.Release()



app = FastAPI(lifespan=lifespan)
app.include_router(studiesRouter)
app.include_router(file_router)
app.include_router(authRouter)

app.add_middleware(
   CORSMiddleware,
    allow_origins = origins,
    allow_credentials =True,
    allow_methods = ["*"],
    allow_headers= ["*"],
)
