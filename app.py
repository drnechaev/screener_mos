"""
Запускаем приложение без веб-сервера
"""

import asyncio
from AiriApp import AiriFMApp

if __name__ == '__main__':

    app = AiriFMApp()
    try:
        asyncio.run(app.Run())
    except KeyboardInterrupt:
        print("Останавливаем приложение")
        app.Release()

