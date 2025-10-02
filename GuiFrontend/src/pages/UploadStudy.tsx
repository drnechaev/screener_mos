import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import { LogOutResponeCheck } from '../auth/auth';

interface FileUploadProps {
  onUploadError?: (error: any) => void;
  acceptedFileTypes?: string;
  maxFileSize?: number; // в байтах
  className?: string;
}

const FileUpload = ({ 
  onUploadError,
  acceptedFileTypes = '.zip,.ZIP',
  maxFileSize = 500 * 1024 * 1024, // 500MB по умолчанию
  className = ''
}: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Валидация файла
  const validateFile = useCallback((file: File): string | null => {
    // Проверка типа файла
    const allowedTypes = ['application/zip', 'application/x-zip-compressed'];
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    
    if (!allowedTypes.includes(file.type) && fileExtension !== 'zip') {
      return 'Пожалуйста, выберите ZIP файл';
    }

    // Проверка размера
    if (file.size > maxFileSize) {
      const maxSizeMB = Math.round(maxFileSize / (1024 * 1024));
      return `Файл слишком большой. Максимальный размер: ${maxSizeMB}MB`;
    }

    // Проверка на пустой файл
    if (file.size === 0) {
      return 'Файл пустой';
    }

    return null;
  }, [maxFileSize]);

  // Реальная загрузка файла через axios
  const uploadFile = useCallback(async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);
    setError(null);
    setFileName(file.name);

    // Создаем FormData
    const formData = new FormData();
    formData.append('uploadFile', file);

    // Создаем AbortController для возможности отмены
    abortControllerRef.current = new AbortController();

    try {
      const response = await axios.post(
        `${import.meta.env.VITE_GUI_BACKEND}/studies/upload`,
        formData,
        {
          signal: abortControllerRef.current.signal,
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            if (progressEvent.total) {
              const progress = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              setUploadProgress(progress);
            }
          },
          timeout: 30 * 60 * 1000, // 30 минут таймаут для больших файлов
        }
      );

      // Загрузка завершена успешно
      setUploadProgress(100);
      
    } catch (error: any) {
      // Обработка ошибок

        LogOutResponeCheck(error)
          
      if (axios.isCancel(error)) {
        setError('Загрузка отменена');
      } else if (error.response) {
        // Ошибка от сервера
        const serverError = error.response.data?.detail || error.response.data?.error || 'Ошибка сервера';
        setError(`Ошибка сервера: ${serverError}`);
        onUploadError?.(error.response.data);
      } else if (error.request) {
        // Ошибка сети
        setError('Ошибка сети. Проверьте подключение к интернету');
        onUploadError?.(error);
      } else {
        // Другие ошибки
        setError(`Ошибка загрузки: ${error.message}`);
        onUploadError?.(error);
      }
    } finally {
      setIsUploading(false);
      abortControllerRef.current = null;
    }
  }, [onUploadError]);

  // Обработчик выбора файла
  const handleFileSelect = useCallback((file: File) => {
    const validationError = validateFile(file);
    
    if (validationError) {
      setError(validationError);
      return;
    }

    uploadFile(file);
  }, [validateFile, uploadFile]);

  // Обработчики drag & drop
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  // Обработчик клика по области загрузки
  const handleAreaClick = useCallback(() => {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  }, [isUploading]);

  // Обработчик изменения input файла
  const handleFileInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  // Отмена загрузки
  const cancelUpload = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    resetUpload();
  }, []);

  // Сброс состояния
  const resetUpload = useCallback(() => {
    setUploadProgress(0);
    setIsUploading(false);
    setError(null);
    setFileName(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  const maxSizeMB = Math.round(maxFileSize / (1024 * 1024));

  return (
    <div className={`flex flex-col items-center space-y-4 p-6 border-2 border-dashed border-gray-300 rounded-lg bg-white ${className} ${
      isDragging ? 'border-blue-500 bg-blue-50' : ''
    }`}>
      {/* Скрытый input файла */}
      <input
        ref={fileInputRef}
        type="file"
        accept={acceptedFileTypes}
        onChange={handleFileInputChange}
        className="hidden"
        disabled={isUploading}
      />

      {/* Область загрузки */}
      <div
        onClick={handleAreaClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`w-full p-8 text-center cursor-pointer transition-all ${
          isDragging ? 'scale-105' : 'hover:bg-gray-50'
        } ${isUploading ? 'cursor-not-allowed opacity-50' : ''}`}
      >
        {/* Иконка */}
        <div className="mx-auto w-16 h-16 mb-4">
          <svg 
            className="w-full h-full text-gray-400" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={1.5} 
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
            />
          </svg>
        </div>

        {/* Текст */}
        <div className="space-y-2">
          <p className="text-lg font-semibold text-gray-700">
            {isUploading ? 'Загрузка...' : 'Загрузите DICOM архив'}
          </p>
          <p className="text-sm text-gray-500">
            {isUploading 
              ? `Загружается: ${fileName}`
              : `Перетащите ZIP архив сюда или нажмите для выбора`
            }
          </p>
          <p className="text-xs text-gray-400">
            Максимальный размер: {maxSizeMB}MB
          </p>
        </div>
      </div>

      {/* Прогресс бар */}
      {isUploading && (
        <div className="w-full space-y-2">
          <div className="flex justify-between text-sm text-gray-600">
            <span>Прогресс загрузки:</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className="bg-blue-500 h-3 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 text-center">
            Не закрывайте страницу до завершения загрузки
          </div>
        </div>
      )}

      {/* Детали файла после успешной загрузки */}
      {fileName && !isUploading && uploadProgress === 100 && (
        <div className="w-full p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center space-x-3">
            <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            <div className="flex-1">
              <p className="font-medium text-green-800">Файл успешно загружен</p>
              <p className="text-sm text-green-600">{fileName}</p>
              <p className="text-xs text-green-500 mt-1">
                Архив отправлен на обработку. Результаты появятся в списке исследований.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Ошибка */}
      {error && (
        <div className="w-full p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-3">
            <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <div className="flex-1">
              <p className="font-medium text-red-800">Ошибка загрузки</p>
              <p className="text-sm text-red-600">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Кнопки действий */}
      <div className="flex space-x-3">
        {isUploading && (
          <button
            onClick={cancelUpload}
            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors font-medium"
            type="button"
          >
            Отменить загрузку
          </button>
        )}
        
        {(uploadProgress === 100 || error) && (
          <button
            onClick={resetUpload}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
            type="button"
          >
            Загрузить другой файл
          </button>
        )}
        
        {!isUploading && uploadProgress === 0 && (
          <button
            onClick={handleAreaClick}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
            type="button"
          >
            Выбрать файл
          </button>
        )}
      </div>

      {/* Информация о поддерживаемых форматах */}
      <div className="text-xs text-gray-400 text-center">
        Поддерживаются ZIP архивы с DICOM файлами. Файлы автоматически обрабатываются после загрузки.
      </div>
    </div>
  );
};

export default FileUpload;