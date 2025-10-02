"""
Класс загрузчика исследований
"""

from typing import Optional, Union, Dict, Tuple, Any
import SimpleITK as sitk
import zipfile
import os
import tempfile
import shutil
import numpy as np
import nibabel as nib

class StudyLoader(object):

    def __init__(self, zip_path:str,output_dir: Optional[str] = None, remove_if_exists: bool = False, 
                 cacheDir: Optional[str] = None, noUnzip:bool = False):

        if not os.path.isfile(zip_path):
            raise FileExistsError(f"File {zip_path} is not exits")

        self._zip_path = zip_path
        self._temp_dir:str = None
        self._series_ids: Tuple = None
        self._currentIter:int = 0
        self._reader = sitk.ImageSeriesReader()
        self._cacheDir = cacheDir

        if output_dir is None:
            self._temp_dir = tempfile.mkdtemp()
            output_dir = self._temp_dir
        else:
            if os.path.isdir(output_dir) and remove_if_exists:
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

        self._output_dir = output_dir

        if not noUnzip:
            self.Unpack()

    def __del__(self):
        print("Destructor")
        self._series_ids = None
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def Unpack(self) -> None:
        try:
            info = self.unpack_zip()
        except Exception as e:
            pass
        print(info)
        self.upload_ct_info()

    def unpack_zip(self) -> Optional[Dict[str, int]]:
        # Распаковываем архив
        total_files = 0
        extracted_files = 0

        print(f"Распаковка архива: {self._zip_path}")
        with zipfile.ZipFile(self._zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            for file_info in file_list:
                # Пропускаем директории
                if file_info.endswith('/'):
                    continue

                total_files += 1

                # Извлекаем только имя файла (без пути)
                filename = os.path.basename(file_info)

                if not filename:
                    continue

                output_path = os.path.join(self._output_dir, filename)

                counter = 1
                while os.path.exists(output_path):
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(self._output_dir, f"{name}_{counter}{ext}")
                    counter += 1

                try:
                    # Извлекаем файл
                    with zip_ref.open(file_info) as source_file:
                        with open(output_path, 'wb') as target_file:
                            target_file.write(source_file.read())
                    extracted_files += 1

                except Exception as e:
                    print(f"Ошибка при извлечении {file_info}: {e}")
                    continue

        return {"total": total_files, "Extraced": extracted_files}

    def SaveResult(self, arr: nib.Nifti1Image, seriesId: str, isMask:bool = False) -> None:
        """Сохраняем результат

        """
        filename = os.path.basename(self._zip_path).replace(".", "_") + "_" + seriesId + ("_mask" if isMask else "") + ".npy"
        if self._cacheDir is not None:
            pathname = self._cacheDir
        else:
            pathname = os.path.abspath(self._zip_path)
        pathname = os.path.join(pathname, filename)
        arr = arr.get_fdata()
        if isMask:
            arr = arr.astype(np.uint8)
        else:
            if arr.max()<=1:
                arr = arr * 65535 - 32768
            arr = arr.astype(np.int16)
        np.save(pathname, arr)

    def LoadResult(self, seriesId: str, isMask:bool = False) -> Optional[np.ndarray]:
        """Функция загрузки результата
        """

        filename = os.path.basename(self._zip_path).replace(".", "_") + "_" + seriesId + ("_mask" if isMask else "") + ".npy"
        if self._cacheDir is not None:
            pathname = self._cacheDir
        else:
            pathname = os.path.abspath(self._zip_path)
        pathname = os.path.join(pathname, filename)
        print("PATH NAME", pathname)
        if os.path.isfile(pathname):
            return np.load(pathname)

        return None

    async def MakeNumpy(self, data:Dict) -> np.ndarray:
        npimg = sitk.GetArrayFromImage(data['series'])
        npimg = np.moveaxis(npimg, (0, 1, 2), (2, 1, 0))
        return (npimg).astype(np.int16)

    def MakeNII(self, data: Dict) -> Any:

        npimg = sitk.GetArrayFromImage(data['series'])
        npimg = np.moveaxis(npimg, (0, 1, 2), (2, 0, 1))
        npimg = (npimg).astype(np.float32)

        affine = data['metadata']['affine']
        LPS2RAS = np.diag([-1, -1, 1, 1])        
        affine = LPS2RAS @ affine
        nii = nib.Nifti1Image(npimg, affine.astype(np.float64, copy=False))
        nii.set_data_dtype(np.float32)

        return nii

    def _get_image_metadata(self, image):
        """
        Извлекает метаданные из DICOM файлов
        """
        metadata = {}

        try:
            # Базовые метаданные изображения
            metadata['size'] = image.GetSize()
            metadata['spacing'] = image.GetSpacing()
            metadata['origin'] = image.GetOrigin()
            metadata['direction'] = image.GetDirection()

            # Преобразуем direction в матрицу 3x3
            direction_matrix = np.array(metadata['direction'])
            affine = np.eye(4)

            if len(direction_matrix)==9:
                direction_matrix = direction_matrix.reshape(3,3)
                affine[:3, :3] = direction_matrix @ np.diag(metadata['spacing'])
                affine[:3, 3] = metadata['origin']
            elif len(direction_matrix)==16:
                direction_matrix = direction_matrix.reshape(4,4)
                affine = direction_matrix @ np.diag(metadata['spacing'])
                affine[:3, 3] = metadata['origin'][:3]


            # Перевод (сдвиг начала координат) 
            metadata['affine'] = affine

            # DICOM метаданные из первого файла
            if self._reader.GetFileNames():
                first_file = self._reader.GetFileNames()[0]
                dicom_reader = sitk.ImageFileReader()
                dicom_reader.SetFileName(first_file)
                dicom_reader.LoadPrivateTagsOn()
                dicom_reader.ReadImageInformation()

                # Извлекаем основные DICOM теги
                tags_to_extract = {
                    'PatientID': '0010|0020',
                    'PatientName': '0010|0010',
                    'StudyDate': '0008|0020',
                    'Modality': '0008|0060',
                    'StudyInstanceUID': '0020|000d',
                    'SeriesInstanceUID': '0020|000e',
                    'SeriesDescription': '0008|103e',
                    'SliceThickness': '0018|0050',
                    'Exposure': '0018|1152',  # Экспозиция
                    'ConvolutionKernel': '0018|1210',  # Ядро свертки
                }
                
                for key, tag in tags_to_extract.items():
                    try:
                        if dicom_reader.HasMetaDataKey(tag):
                            metadata[key] = dicom_reader.GetMetaData(tag).strip()
                    except:
                        metadata[key] = 'Не доступно'
                        
        except Exception as e:
            print(f"Ошибка при извлечении метаданных: {e}")
        
        return metadata

    def upload_ct_info(self) -> None:
        """
        Загружает КТ исследование из zip-архива
        """

        try:       
            # Получаем список серий
            self._series_ids = self._reader.GetGDCMSeriesIDs(self._output_dir)

            if not self._series_ids:
                raise ValueError("Не удалось идентифицировать DICOM серии")
            print(self._series_ids)

            print(f"Найдено серий: {len(self._series_ids)}: {self._series_ids}")

        except Exception as e:
            print(f"Ошибка при загрузке КТ исследования: {e}")
            raise

    def upload_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        # Выбираем серию
        if series_id not in self._series_ids:
            return None
        # Получаем файлы для выбранной серии
        dicom_names = self._reader.GetGDCMSeriesFileNames(self._output_dir, series_id)

        if not dicom_names:
            raise ValueError(f"Не удалось получить файлы для серии {series_id}")

        # Устанавливаем файлы для чтения
        self._reader.SetFileNames(dicom_names)

        # Настраиваем параметры чтения (опционально)
        self._reader.SetOutputPixelType(sitk.sitkInt16)  # Тип пикселей для КТ

        # Загружаем изображение
        print("Загрузка КТ изображения...")
        image = self._reader.Execute()

        size = image.GetSize()

        dims_to_remove = [i for i, s in enumerate(size) if s == 1]

        if dims_to_remove:
            # Удаляем первую найденную ось с размером 1
            dim_to_remove = dims_to_remove[0]
            extract_filter = sitk.ExtractImageFilter()

            new_size = list(size)
            new_index = [0] * image.GetDimension()

            new_size[dim_to_remove] = 0
            new_index[dim_to_remove] = 0

            extract_filter.SetSize(new_size)
            extract_filter.SetIndex(new_index)
            image = extract_filter.Execute(image)

        # Получаем метаданные
        metadata = self._get_image_metadata(image)

        print("КТ исследование успешно загружено!")
        print(f"Размер изображения: {image.GetSize()}")
        print(f"Пространственное разрешение: {image.GetSpacing()}")

        return {"series": image, "metadata": metadata, "series_id": series_id}

    def __getitem__(self, id: Union[str, tuple]) -> Tuple[sitk.Image, Dict[str, Any]]:
        if isinstance(id, int):
            id = self._series_ids[id]

        if not isinstance(id, str):
            raise ValueError("Index must be type of str or int")

        return self.upload_series(id)

    def __iter__(self):
        self._currentIter = 0
        return self

    def __next__(self):
        if self._currentIter < len(self._series_ids):
            result = self[self._currentIter]
            self._currentIter += 1
            return result
        else:
            raise StopIteration

    @property
    def SeriesIDS(self) -> Tuple[str, ...]:
        return self._series_ids