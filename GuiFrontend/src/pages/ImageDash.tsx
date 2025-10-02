import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { IImageContent } from './workspace'; // Adjust import path as needed

interface ImageData {
  image: number[][][]; // uint16 numpy array (w, h, d)
  mask: number[][][] | null; // int16 numpy array (w, h, d)
}

const ImageDash = ({ imageContent }: { imageContent: IImageContent }) => {
  const [imageData, setImageData] = useState<ImageData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [currentSlice, setCurrentSlice] = useState<number>(0);
  const [showMask, setShowMask] = useState<boolean>(false);
  const [windowType, setWindowType] = useState<'lung' | 'soft-tissue'>('lung');
  const canvasRef = useRef<HTMLCanvasElement>(null);


    console.log(imageContent)

  // Window level presets
  const windowPresets = {
    lung: { level: -500, width: 1500 },
    'soft-tissue': { level: 50, width: 350 }
  };

  // Load image data
  useEffect(() => {

    const fetchImageData = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await axios.get<ImageData>(
          `${import.meta.env.VITE_GUI_BACKEND}/studies/${imageContent.study_name}/${imageContent.series_id}`, 
            {
              withCredentials: true,
              headers: {
              'Content-Type': 'application/json',
              }
            }
          );
        console.log(response.data)
        setImageData(response.data);
      } catch (err) {
        setError('Failed to load image data');
        console.error('Error loading image data:', err);
      } finally {
        setLoading(false);
      }
    };
    if (imageContent.study_name && imageContent.series_id)
          fetchImageData();
  }, [imageContent]);

  // Convert uint16 to uint8 with windowing
  const applyWindowing = useCallback((pixelValue: number, windowLevel: number, windowWidth: number): number => {
    const windowMin = windowLevel - windowWidth / 2;
    const windowMax = windowLevel + windowWidth / 2;
    
    let value = ((pixelValue - windowMin) / windowWidth) * 255;
    value = Math.max(0, Math.min(255, value));
    return Math.round(value);
  }, []);

  // Check if mask pixel should be displayed (non-zero values)
  const shouldDisplayMask = useCallback((maskValue: number): boolean => {
    return maskValue !== 0;
  }, []);

  // Render image and mask
  useEffect(() => {
    if (!imageData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { image, mask } = imageData;
    
    if (!image.length || !image[0]?.length) return;
    
    const width = image.length;
    const height = image[0].length;
    const depth = image[0][0]?.length || 1;
    
    if (currentSlice >= depth) {
      setCurrentSlice(Math.max(0, depth - 1));
      return;
    }

    canvas.width = width;
    canvas.height = height;

    const imageDataBuffer = ctx.createImageData(width, height);
    const preset = windowPresets[windowType];

    // Render image
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelValue = image[x]?.[y]?.[currentSlice] || 0;
        const windowedValue = applyWindowing(pixelValue, preset.level, preset.width);
        
        const idx = (y * width + x) * 4;
        imageDataBuffer.data[idx] = windowedValue;     // R
        imageDataBuffer.data[idx + 1] = windowedValue; // G
        imageDataBuffer.data[idx + 2] = windowedValue; // B
        imageDataBuffer.data[idx + 3] = 255;           // A
      }
    }

    ctx.putImageData(imageDataBuffer, 0, 0);

    // Render mask if enabled and exists
    if (showMask && mask && mask.length > 0) {
      // Create a temporary canvas for mask compositing
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = width;
      maskCanvas.height = height;
      const maskCtx = maskCanvas.getContext('2d');
      
      if (maskCtx) {
        const maskDataBuffer = maskCtx.createImageData(width, height);
        
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const maskValue = mask[x]?.[y]?.[currentSlice] || 0;
            
            if (shouldDisplayMask(maskValue)) {
              const idx = (y * width + x) * 4;
              // Red color for mask with some transparency
              maskDataBuffer.data[idx] = 255;     // R
              maskDataBuffer.data[idx + 1] = 0;   // G
              maskDataBuffer.data[idx + 2] = 0;   // B
              maskDataBuffer.data[idx + 3] = 100; // A (semi-transparent)
            }
          }
        }
        
        maskCtx.putImageData(maskDataBuffer, 0, 0);
        // Composite mask over the original image
        ctx.drawImage(maskCanvas, 0, 0);
      }
    }
  }, [imageData, currentSlice, showMask, windowType, applyWindowing, shouldDisplayMask]);

  // Keyboard and wheel events for slice navigation
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (!imageData || !imageData.image.length) return;
      
      const depth = imageData.image[0]?.[0]?.length || 1;
      switch (e.key) {
        case 'ArrowUp':
        case 'PageUp':
          setCurrentSlice(prev => Math.min(prev + 1, depth - 1));
          break;
        case 'ArrowDown':
        case 'PageDown':
          setCurrentSlice(prev => Math.max(prev - 1, 0));
          break;
        case 'Home':
          setCurrentSlice(0);
          break;
        case 'End':
          setCurrentSlice(depth - 1);
          break;
        default:
          break;
      }
    };

    const handleWheel = (e: WheelEvent) => {
      if (!imageData || !imageData.image.length) return;
      
      e.preventDefault();
      const depth = imageData.image[0]?.[0]?.length || 1;
      setCurrentSlice(prev => {
        if (e.deltaY > 0) {
          return Math.max(prev - 1, 0);
        } else {
          return Math.min(prev + 1, depth - 1);
        }
      });
    };

    const canvas = canvasRef.current;
    if (canvas) {
      canvas.addEventListener('wheel', handleWheel, { passive: false });
    }
    window.addEventListener('keydown', handleKeyPress);

    return () => {
      if (canvas) {
        canvas.removeEventListener('wheel', handleWheel);
      }
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [imageData]);

  const handleWindowTypeChange = (type: 'lung' | 'soft-tissue') => {
    setWindowType(type);
  };

  const handleMaskToggle = () => {
    setShowMask(!showMask);
  };

  const handleSliceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentSlice(Number(e.target.value));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-xl font-semibold">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-xl font-semibold text-red-600">{error}</div>
      </div>
    );
  }

  if (!imageData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-xl font-semibold">No image data available</div>
      </div>
    );
  }

  const depth = imageData.image[0]?.[0]?.length || 1;
  const hasMask = imageData.mask && imageData.mask.length > 0;

  return (
    <div className="flex flex-col items-center p-4 bg-gray-100 rounded-lg">
      <div className="flex flex-wrap gap-4 mb-4 justify-center">
        <button
          onClick={() => handleWindowTypeChange('lung')}
          className={`px-4 py-2 rounded transition-colors ${
            windowType === 'lung' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-300 hover:bg-gray-400'
          }`}
        >
          Легочное окно
        </button>
        <button
          onClick={() => handleWindowTypeChange('soft-tissue')}
          className={`px-4 py-2 rounded transition-colors ${
            windowType === 'soft-tissue' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-300 hover:bg-gray-400'
          }`}
        >
          Мягкотканное окно
        </button>
        <button
          onClick={handleMaskToggle}
          disabled={!hasMask}
          className={`px-4 py-2 rounded transition-colors ${
            showMask 
              ? 'bg-green-600 text-white' 
              : 'bg-gray-300 hover:bg-gray-400'
          } ${!hasMask ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {showMask ? 'Скрыть маску' : 'Показать маску'}
        </button>
      </div>

      <div className="relative border-2 border-gray-300 rounded-lg overflow-hidden bg-black">
        <canvas
          ref={canvasRef}
          className="max-w-full max-h-screen cursor-crosshair"
        />
      </div>

      <div className="flex items-center gap-4 mt-4 w-full max-w-2xl">
        <input
          type="range"
          min="0"
          max={Math.max(0, depth - 1)}
          value={currentSlice}
          onChange={handleSliceChange}
          className="flex-1"
        />
        <span className="font-mono text-lg min-w-[120px]">
          Срез: {currentSlice + 1} / {depth}
        </span>
      </div>

      <div className="mt-4 text-sm text-gray-600 text-center">
        Используйте колесо мыши или стрелки вверх/вниз для навигации по слоям
        <br />
        {hasMask ? 'Маска доступна' : 'Маска не доступна'}
      </div>
    </div>
  );
};

export default ImageDash;