from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Request
from fastapi.responses import JSONResponse
import os
import aiofiles

# Конфигурация

SECRET_TOKEN = "MOSCOWTOCKEN2025"  # Зашитый токен


file_router = APIRouter(prefix='/v1')

@file_router.post("/upload")
async def upload_file(
    request: Request,
    token: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Загружает файл если токен верный
    """

    if token != SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    dataPath = request.state.app.Studies.DataPath
    file_name = file.filename

    try:
        if file_name.endswith(".zip"):
            i = 1
            while os.path.isfile(os.path.join(dataPath, file_name)):
                file_name = file_name.split(".")[0] + f"_{i:03d}.zip"
                i += 1
            request.state.loger.debug(f"Save file: {file_name}")

            file_path = os.path.join(dataPath, file_name)

            contents = await file.read()
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(contents)

            await request.state.app.Studies.AddData(file_name, True)
            return JSONResponse(content={"Status":"ok", "StudyID": file_name}, status_code=200)
        else:
            request.state.loger.error("Прислан не верный формат данных")
            return JSONResponse(content={"Status":"error","msg":"Bad file format"}, status_code=400)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )


@file_router.post("/download")
async def download_excel(token: str, request: Request):
    file_path = request.state.app.Studies.OutputXlsPath

    if token != SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found!")

    return FileResponse(
        path=file_path,
        filename=file_path.split("/")[-1],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
