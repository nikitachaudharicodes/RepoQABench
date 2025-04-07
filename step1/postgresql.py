import asyncpg
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_db():
    try:
        conn = await asyncpg.connect(
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        print("Connected to PostgreSQL successfully!")
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test_db())
