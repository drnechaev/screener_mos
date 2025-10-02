"""
Класс хранения данных об загруженных и обработанных исследованиях
"""

from typing import Union, Optional
import zipfile
import shutil
import os
import logging
import numpy as np

import pandas as pd

DATA_NII = 0x01
DATA_NDARRAY = 0x02


class StudiesClass: ...

class StudiesClass(object):

    def __init__(self, dataFolderPath: str = "./Data/data", outFolderPath:str = "./Data/output",
                    pdPath: str = "./Data/", logName:str = 'AiriFM') -> None:
        """Констуктор

        """

        self._logger = logging.getLogger(logName)
        self._isZip: bool = False

        if dataFolderPath.endswith("zip"):
            self._logger.info(f"Директория с даннымми {dataFolderPath} архив")
            self._logger.info("Распаковываем ...")

            self._dataFolderPath = dataFolderPath.replace[:-4]
            os.makedirs(self._dataFolderPath, exist_ok=True)

            try:
                # Распаковываем архив
                with zipfile.ZipFile(dataFolderPath, 'r') as zip_ref:
                    zip_ref.extractall(self._dataFolderPath)
                print(f"Файлы распакованы в: {self._dataFolderPath}")
                self._isZip = True
            except Exception as e:
                # В случае ошибки удаляем временную директорию
                shutil.rmtree(self._dataFolderPath)
                raise e
        else:
            if not os.path.isdir(dataFolderPath):
                os.makedirs(dataFolderPath)
            self._dataFolderPath = dataFolderPath

        self._logger.debug(f"Директория с данными {self._dataFolderPath}")

        if not os.path.isdir(outFolderPath):
            os.makedirs(outFolderPath)
        self._outFolderPath = outFolderPath
        self._logger.debug(f"Директория для результатов {self._outFolderPath}")

        self._dfPath = os.path.join(pdPath, "airi.csv")

        if not os.path.isfile(self._dfPath):
            self._df = pd.DataFrame(columns=['studyName', 'path_to_study', 'study_uid', 'series_uid',
                'probability_of_pathology', 'pathology', 'processing_status',
                'time_of_processing', 'most_dangerous_pathology_type', 'pathology_localization',
                'debug_message', 'proccessed', 'is_return'])
            self._save()
        else:
            self._logger.debug("Загружаем уже существующую базу")
            self._df = pd.read_csv(self._dfPath)

        self._NewData: set = set()

        for _p in os.listdir(self._dataFolderPath):
            if not _p.endswith(".npy"):
                self._AddData(_p, True)

        self._NewData = self._NewData.union(self._df[self._df['proccessed']==False]['studyName'].to_list())
        if len(self._NewData):
            self._logger.debug(f"Не обработанные исследования на момент запуска: {','.join(self._NewData)}")
            self._save()

    def Release(self) -> None:
        """Освобождаем ресурсы
        """
        del self._df
        del self._NewData

    def Export(self) -> None:
        self._df[self._df['is_return']==True][['path_to_study', 'study_uid', 'series_uid', 'probability_of_pathology', 'pathology',
                    'processing_status', 'time_of_processing','most_dangerous_pathology_type', 
                    'pathology_localization']].to_excel(os.path.join(self._outFolderPath, "result.xlsx"), index=False)

    def _save(self) -> None:
        self._df.to_csv(self._dfPath, index=False)


    def _AddData(self, id: str, isReturn:bool = False) -> bool:
        """Добавляем данные в базу
        """

        if id not in self._df['studyName'].to_list():
            self._df.loc[len(self._df)] = {'studyName': id,
                'path_to_study': os.path.join(self._dataFolderPath, id),
                'study_uid': "",
                'series_uid': "",
                'probability_of_pathology': None,
                'pathology': None,
                'processing_status': "",
                'time_of_processing': None,
                'most_dangerous_pathology_type': "",
                'pathology_localization': "",
                'debug_message': "",
                'proccessed': False, 
                'is_return': isReturn
                }

            self._NewData.add(id)

            return True

        return False

    async def AddData(self, path: str, isReturn:bool = False) -> None:
        """Асинхронная обертка над

        Returns:
            bool: True если успешно
        """
        if self._AddData(path, isReturn):
            self._save()

    async def HasNew(self) -> Union[str, None]:
        """Возращает случайное исследование из списка добавленных исследований, 
            при этом данное исследование из данного списка удалется

        Returns:
            Union[str,None]: идентификатор исследования или None
        """
        if len(self._NewData)!=0:
            return self._NewData.pop()

        return None

    async def GetData(self, id:str, seriesId:Optional[str] = None) -> Union[dict, None]:
        """Получаем строку по id
        """

        series = self._df[self._df['studyName']==id]
        if seriesId is not None:
            seriesWithID = series[series['series_uid']==seriesId]
            if len(seriesWithID)!=0:
                return seriesWithID.to_dict('records')
            else:
                return None

        if len(series):
            return series.to_dict('records')

        return None

    def UpdateData(self, study_id:str, data:dict) -> None:

        series_uid = data.get('series_uid')

        # Удаляем строку с пустым series_uid для этого study_id
        empty_series_mask = (self._df['studyName'] == study_id) & (self._df['series_uid'].isna() | (self._df['series_uid'] == ''))
        self._df = self._df[~empty_series_mask]

        if series_uid:  # Если series_uid указан
            # Ищем существующую запись с таким series_uid
            existing_mask = (self._df['studyName'] == study_id) & (self._df['series_uid'] == series_uid)

            if existing_mask.any():
                # Обновляем существующую запись
                update_cols = [col for col in data if col in self._df.columns]
                self._df.loc[existing_mask, update_cols] = [data[col] for col in update_cols]
            else:
                # Создаем новую запись
                self._df = pd.concat([self._df, pd.DataFrame([data])], ignore_index=True)
        else:
            # Создаем новую запись без series_uid
            self._df = pd.concat([self._df, pd.DataFrame([data])], ignore_index=True)

        self._save()   


    async def GetDatas(self) -> Union[list, None]:
        """Получаем все id исследований
        """

        df_clean = self._df.replace([np.nan, -np.inf, np.inf], None)
        return df_clean.to_dict("records")

    async def DeleteData(self, id:str) -> bool:
        """Удаляем строку из таблицы по id
        """

        return True

    @property
    def DataPath(self) -> str:
        return self._dataFolderPath

    @property
    def OutputXlsPath(self) -> str:
        return os.path.abspath(os.path.join(self._outFolderPath, "result.xlsx"))