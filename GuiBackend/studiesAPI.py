from fastapi import APIRouter, Depends, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from .auth import GetCurrentUser
import os

from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import aiofiles

import numpy as np


studiesRouter = APIRouter(prefix="/studies", dependencies=[Depends(GetCurrentUser)])


@studiesRouter.get("/")
async def studies(request: Request) -> JSONResponse:
    """Возращает список обработанных исследований из базы без суффикса thumb
    """

    mainAPP = request.state.app

    return JSONResponse(content={"status":" ok", "Studies": await mainAPP.Studies.GetDatas()}, 
                        status_code=200, 
                        media_type="application/json")


@studiesRouter.get("/download/")
async def download_excel(request: Request):
    file_path = request.state.app.Studies.OutputXlsPath

    print("DS", file_path)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found!")

    return FileResponse(
        path=file_path,
        filename=file_path.split("/")[-1],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@studiesRouter.get("/{StudyName}/{SeriesID}")
async def study(StudyName:str, SeriesID:str, request: Request) -> JSONResponse:
    """Возращает список обработанных исследований из базы без суффикса thumb

    """
    loader = None
    try:

        request.state.loger.debug(f"Get Study Response: StudyName={StudyName}, SeriesID={SeriesID}")
        study = await request.state.app.Studies.GetData(StudyName, SeriesID)
        if study is None:
            raise
        study = study[0]
        request.state.loger.debug(f"Study {study}")

        loader = await request.state.app.CreateLoader(study['path_to_study'], noUnzip=True)

        mask = loader.LoadResult(SeriesID, True)
        if mask is None:
            print("Mask is nont")
            loader.Unpack()
            series = loader[SeriesID]
            if series is None:
                raise

            img = await loader.MakeNumpy(series)
        else:
            img = loader.LoadResult(SeriesID, False)
        

        # Создаем ответ в нужном формате, конвертируя numpy arrays в списки
        response_data = {
            "image": img.tolist(),  # Конвертируем в list
            "mask": mask.tolist() if mask is not None else None   # Конвертируем в list
        }

        return JSONResponse(
            content=response_data,
            status_code=200
        )

    except Exception as e:
        request.state.loger.exception(f"Error generating study data: {str(e)}")
        return JSONResponse(
            content={"status": "error", "msg": f"Study not found: {str(e)}"},
            status_code=404
        )
    finally:
        if loader is not None:
            del loader

@studiesRouter.post("/upload")
@studiesRouter.post("/upload/")
async def uploadStudy(uploadFile: UploadFile, request: Request) -> JSONResponse:
    """Функция приема файла из POST запроса и отправки
    в модель
    """

    request.state.loger.debug(f"FORM DATA: {uploadFile.filename}")

    dataPath = request.state.app.Studies.DataPath
    file_name = uploadFile.filename

    if uploadFile.filename.endswith(".zip"):
        i = 1
        while os.path.isfile(os.path.join(dataPath, file_name)):
            file_name = file_name.split(".")[0] + f"_{i:03d}.zip"
            i += 1
        request.state.loger.debug(f"Save file: {file_name}")

        file_path = os.path.join(dataPath, file_name)

        contents = await uploadFile.read()
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(contents)
    else:
        request.state.loger.error("Прислан не верный формат данных")
        return JSONResponse(content={"Status":"error","msg":"Bad file format"}, status_code=400)

    #studyID = f"Uploaded{uploadFile.filename.replace('_','')}"
    await request.state.app.Studies.AddData(file_name, True)
    #GetApp().Studies._AddData(id=studyID,Modality=modality,Datasets=[dcm], isReturn=False, dataFormat=studyFormat)


    return JSONResponse(content={"Status":"ok", "StudyID": file_name}, status_code=200)
