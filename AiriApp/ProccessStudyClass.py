"""
Обработка исследования моделью

"""

from typing import Dict, Optional
from datetime import datetime

from torch import nn

from .StudiesClass import DATA_NII, DATA_NDARRAY
from . import StudiesClass


class ProccessStudyClass(object):

    def __init__(self, config:Dict, studies:StudiesClass) -> None:
        """Создаем класс модели определения плоскостопия

        Args:
            config (ConfigClass): объект данных конфигурации
            studies:StudiesClass: Объект класса исследований
        """
        self._config = config
        # _boneModel = resnet50(0.39,nn.GELU())

        self._model = None

        self._studies = studies


    async def ProccessStudy(self, studyId:str, studyData:dict) -> Optional[Dict]:
        """Обработка исследование из базы (studyData)
        Обрабтывается, только если флаг в данных Proccessed установлен в False
        Если формат данных DATA_DICOM то вначале проверяем, соотвествует ли исследование критериям(фильтру)
        После обработки всех изображений исследований, добававляет обработанные изображения в базу с тем же идентификатором
            и суфиктом thumb. ("STUDYID" -> "STUDYIDthumb"). Формат соответсвует исходному DATA_FORMAT. Флаг Proccessed 
            устанавливается в True

        Args:
            studyId (str): идентификатор исследования
            studyData (dict): данные исследования

        Returns:
            Union[list[Dataset], None]: возращает список обработанных исследование в том же формате, что и DataFormat
        """

        proccessedDataSet = {"dcm":[]}
        log_debug("Обрабатываем исследование:", studyId)
        #currentStudies = await self._studies.GetData(studyId)

        if studyData['Proccessed']:
            log_debug(f"Исследование {studyId} уже обработанное")
            return studyData['Data']

        if all(mdx not in studyData['Modalities'] for mdx in ['DX',"CR"]):
            log_debug(f"Модальность исследования {studyId} не поддерживается")
            return None

        # thumbId = studyId+"thumb"
        # if studyData['DataFormat']== DATA_NII:
        #     SeriesInstanceUID = pydicom.uid.generate_uid()  # Генерация нового UID для изображения
        #     SOPInstanceUID = pydicom.uid.generate_uid()  # Генерация нового UID для изображения

        lastSendIndex = 1

        creationDate = datetime.now().strftime("%Y%m%d")
        creationTime = datetime.now().strftime("%H%M%S.%f")

        for dcm in studyData['Data']:
            if studyData['DataFormat']==DATA_DICOM:
                if not self._model.flatfeetFrontFilter(dcm):
                    log_info("Исследование Study:",dcm.StudyInstanceUID, "Series:", dcm.SeriesInstanceUID,
                              "SOP:", dcm.SOPInstanceUID,"не прошло фильтрацию")
                    continue
                if dcm.PhotometricInterpretation not in ('MONOCHROME1','MONOCHROME2'):
                    log_warn("Формат исследования Study:",dcm.StudyInstanceUID, "Series:", dcm.SeriesInstanceUID,
                              "SOP:", dcm.SOPInstanceUID,"Format:",dcm.PhotometricInterpretation)
                    continue

                image, sparsing = self._model.uploadDCM(dcm)
                log_debug("Запускаем модель для файла", dcm.SOPInstanceUID)
            elif studyData['DataFormat']==DATA_NDARRAY:
                image = dcm
                sparsing = [1.0,1.0]
                log_debug("Запускаем модель для файла изображения", studyId)

            img, _, _, _, _ = self._model.predict(image,True,sparcing=sparsing)

            if studyData['DataFormat']==DATA_DICOM:
                if 0x00280106 in dcm:
                    del dcm[0x0028, 0x0106]
                if 0x00280107 in dcm:
                    del dcm[0x0028, 0x0107]
                    
                if 0x00180024 in dcm:
                    del dcm[0x0018,0x0024]
                if 0x00181030 in dcm:
                    del dcm[0x0018,0x1030]
                    
                dcm.SOPClassUID = SecondaryCaptureImageStorage  # Меняем класс не все PACS поддерживают
                dcm.SOPInstanceUID = SOPInstanceUID[:-2] + "." + str(lastSendIndex)
                dcm.SeriesDate = creationDate
                dcm.ContentDate = creationDate #date of content
                dcm.SeriesTime = creationTime
                dcm.ContentTime = creationTime   #date of time
                dcm.InstanceNumber = lastSendIndex
                dcm.SeriesInstanceUID = SeriesInstanceUID
                dcm.SeriesNumber = 1001
                dcm.SeriesDescription = "AiriFM"
                dcm.SamplesPerPixel = 3
                dcm.PhotometricInterpretation = "RGB"
                dcm.BitsStored = 8
                dcm.BitsAllocated = 8
                dcm.HighBit = 7
                dcm.PixelRepresentation = 0
                dcm.WindowCenter = 127
                dcm.WindowWidth = 255
                dcm.PlanarConfiguration = 0
                dcm.PixelData = img.tobytes()

                proccessedDataSet['dcm'].append(dcm)
                
                lastSendIndex += 1
            else:
                proccessedDataSet['dcm'].append(img)

                
        await self._studies.DeleteData(studyId)
        log_debug("LEN of dataas",len(proccessedDataSet['dcm']))
        await self._studies.AddData(thumbId,studyData['Modalities'],proccessedDataSet['dcm'],True,False,studyData['DataFormat'])
        
                
        return proccessedDataSet['dcm']
