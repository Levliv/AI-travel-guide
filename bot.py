from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import httpx
import os

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = "http://localhost:8001"

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Отправь вопрос про путешествия.")

@dp.message()
async def handle_query(message: types.Message):
    print("="*25 + message.text + "="*25 )
    async with httpx.AsyncClient(timeout=60.0) as client:  # увеличь timeout
        try:
            resp = await client.post(f"{API_URL}/answer", json={"text": message.text})
            data = resp.json()
            print(data["answer"])
            await message.answer(str(data["answer"]))
        except httpx.ReadTimeout:
            print('TimeOut')
            await message.answer("API request timeout")
        except Exception as e:
            print('Exception')
            await message.answer(f"Error: {str(e)}")

import asyncio
print('Startup finished')
asyncio.run(dp.start_polling(bot))
