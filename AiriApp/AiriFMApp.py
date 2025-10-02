"""
Класс приложения
"""

from typing import Dict
import os
import sys
from .StudiesClass import StudiesClass
from .StudyLoader import StudyLoader
from .ScreenerModel import ScreenerModel

import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import logging
import json
import time

from logging.handlers import RotatingFileHandler

class AiriFMApp(object):
    """Класс приложения определения норма ы и патологии на КТ ОГК

    """

    def __init__(self, saveMask:bool = True) -> None:
        """
        Конструктор
        Тут мы загружаем файл конфигурации, создает класс хранящий исследования в работе
        """

        self._saveMask = saveMask

        if not "CONFIG" in os.environ:
            os.environ['CONFIG'] = './config.json'
        print("Используем конфигурационный файл:", os.environ['CONFIG'])
        try:
            with open(os.environ['CONFIG'], "r") as f:
                self._config = json.load(f)
        except:
            self._config = {}

        self._cachedir = self._config.get("CacheDir", "./Data/cache")
        if self._cachedir is not None:
            if not os.path.isdir(self._cachedir):
                os.makedirs(self._cachedir)

        self._logName: str = "AiriFM"
        logFormat: str = "%(name)s: %(asctime)s [%(levelname)s] %(message)s"

        logPath: str = "./Data/logs"
        logFile: str = "logfile.log"

        file_handeler = RotatingFileHandler(
                os.path.join(logPath, logFile), mode="a", maxBytes=1024 * 1024 * 10, backupCount=100, encoding="utf-8"
            )
        log_format = logging.Formatter(logFormat)
        file_handeler.setFormatter(log_format)
        self._loger = logging.getLogger(self._logName)
        self._loger.addHandler(file_handeler)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(log_format)
        self._loger.addHandler(consoleHandler)

        self._loger.setLevel(logging.DEBUG)
        self._loger.info("Создаем класс исследований")
        self._studies:StudiesClass = StudiesClass(dataFolderPath=self._config.get("DataPath", "./Data/data"),
                                                    outFolderPath=self._config.get("OutputPath","./Data/output"))

        self._model = ScreenerModel(self._config.get("checkpoint", "./Data/Weights/glow_model.pt"))

        self._runAPP = True

        self._loger.info("Класс приложения создан")

    async def CreateLoader(self, studyPath, noUnzip:bool=False) -> StudyLoader:
        return StudyLoader(studyPath, cacheDir=self._cachedir, noUnzip=noUnzip)

    async def Run(self) -> None:
        """Вечный цикл приложения проверяющий, есть ли новые исследования. 
        Если исследование есть, оно отправляется на обработку в модель.
        Если установлен флаг IsReturn, то отправлеяем обработанные исследования во внешний PACS
        """

        self._loger.info("Запускаем цикл")

        while self._runAPP:

            studyId = await self._studies.HasNew()
            if studyId is not None:
                studyData = await self._studies.GetData(studyId)
                studyPath = studyData[0]['path_to_study']
                self._loger.debug(f"Обрабатываем исследование {studyPath}")

                try:
                    loader = StudyLoader(studyPath, cacheDir=self._cachedir)

                    for _data in loader:

                        if len(studyData) != 1:
                            studyData = await self._studies.GetData(studyId, _data['series_id'])

                        studyData = studyData[0]

                        print(_data['metadata'])
                        if _data['metadata']['Modality'] == 'CT':

                            nii = await asyncio.get_running_loop().run_in_executor(None, loader.MakeNII, _data)
                            self._loger.debug("Получили Nii")
                            start_time = time.time()
                            self._loger.debug("Отправляем в модель")

                            mp_context = get_context('spawn')

                            with ProcessPoolExecutor(
                                    max_workers=1,
                                    mp_context=mp_context
                                ) as executor:
                                result = await asyncio.get_running_loop().run_in_executor(executor, self._model.predict, nii,
                                                        _data['metadata']['StudyInstanceUID'], 
                                                        _data['metadata']['SeriesInstanceUID'])

                            # result = ({'study_uid': '1.2.276.0.7230010.3.1.2.2462171185.19116.1754559949.863', 'series_uid': '1.2.276.0.7230010.3.1.3.2462171185.19116.1754559949.864', 'probability_of_pathology': 0.475800242442684, 'pathology': 0, 'processing_status': 'Success', 'time_of_processing': None, 'most_dangerous_pathology_type': '', 'pathology_localization': None, 'debug_message': ''},
                            #           nii, None, nii, None)

                            execution_time = time.time() - start_time
                            self._loger.info(result)
                        studyData.update(result[0])
                        studyData['proccessed'] = True
                        studyData['time_of_processing'] = int(execution_time)
                        self._studies.UpdateData(studyId, studyData)

                        if result[2] is not None:
                            self._loger.debug("Сохраняем результаты")
                            loader.SaveResult(result[2], _data['series_id'], False)
                            self._loger.debug("Сохранили")

                        if result[3] is not None:
                            self._loger.debug("Сохраняем маску")
                            loader.SaveResult(result[3], _data['series_id'], True)
                            self._loger.info(f"Маска для серии {_data['series_id']} сохранена")

                        if studyData['is_return']:
                            self._studies.Export()
                            self._loger.info("Результаты экспортированны")

                    self._loger.debug(f"Исследование {studyData['path_to_study']} обработано")
                    del loader
                except Exception as e:
                    if isinstance(studyData, list):
                        studyData = studyData[0]
                    studyData['proccessed'] = True
                    studyData['processing_status'] = 'Failure'
                    studyData['debug message'] = f'Error during executing\n {str(e)}'
                    self._studies.UpdateData(studyId, studyData)

            await asyncio.sleep(0.5)

    def Release(self) -> None:
        """Освобождение ресурсов, не асинхронная функция
        """
        self._runAPP = False
        self._studies.Release()

    @property
    def Studies(self) -> StudiesClass:
        return self._studies


    @property
    def Config(self) -> Dict:
        return self._config
