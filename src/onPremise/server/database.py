import sqlite3 
import aiosqlite 
from typing import Dict, List, Optional 
from datetime import datetime

class Database: 
  def __init__(self, db_path="db.db"): 
    self.db_path = db_path
    self.init_db() 

  def init_db(self): 
    with sqlite3.connect(self.db_path) as conn:
      # Add IF NOT EXISTS to prevent errors
      conn.executescript("""
          CREATE TABLE IF NOT EXISTS rooms (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL
          );

          CREATE TABLE IF NOT EXISTS cameras (
              id TEXT PRIMARY KEY,
              name TEXT,
              room_id INTEGER,
              status TEXT,
              last_seen TIMESTAMP,
              FOREIGN KEY (room_id) REFERENCES rooms(id)
          );

          CREATE TABLE IF NOT EXISTS alarm (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              camera_id TEXT,
              room_id INTEGER,
              timestamp TIMESTAMP,
              active BOOLEAN,
              FOREIGN KEY (camera_id) REFERENCES cameras(id)
              FOREIGN KEY (room_id) REFERENCES rooms(id)
          );
      """)


  async def create_camera(self, camera_id: str, name: str, status: str):
    async with aiosqlite.connect(self.db_path) as db:
      await db.execute("""
          INSERT INTO cameras (id, name, status)
          VALUES (?, ?, ?)
      """, (camera_id, name, status))
      await db.commit()

  async def delete_camera(self, camera_id: str):
    async with aiosqlite.connect(self.db_path) as db:
      await db.execute("""
          DELETE FROM cameras
          WHERE id = ?
      """, (camera_id,))
      await db.commit()

  async def get_cameras(self) -> List[dict]: 
    async with aiosqlite.connect(self.db_path) as db: 
      db.row_factory = aiosqlite.Row
      async with db.execute("SELECT * FROM cameras") as cursor: 
        return [dict(row) for row in await cursor.fetchall()]

  async def get_camera(self, camera_id: str) -> Optional[Dict]:
    async with aiosqlite.connect(self.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


  async def update_camera_status(self, camera_id: str, status: str):  
    async with aiosqlite.connect(self.db_path) as db:
      await db.execute("""
          UPDATE cameras 
          SET last_seen = ?, status = ?
          WHERE id = ?
      """, (datetime.now(), status, camera_id))
      await db.commit() 
    
  async def get_rooms(self) -> List[dict]: 
    async with aiosqlite.connect(self.db_path) as db: 
      db.row_factory = aiosqlite.Row
      async with db.execute("SELECT * FROM rooms") as cursor: 
        return [dict(row) for row in await cursor.fetchall()]
      
  async def update_camera(self, camera_id: str, data: dict): 
    async with aiosqlite.connect(self.db_path) as db:
      await db.execute("""
          UPDATE cameras 
          SET last_seen = ?, status = ?
          WHERE id = ?
      """, (datetime.now(), data["status"], camera_id))
      await db.commit()

  async def create_room(self, room_name: str):
    async with aiosqlite.connect(self.db_path) as db:
      await db.execute("""
          INSERT INTO rooms (name)
          VALUES (?, ?)
      """, (room_name))
      await db.commit()

  async def assign_camera_room(self, camera_id: int, room_id: int):
    async with aiosqlite.connect(self.db_path) as db:
      await db.execute("""
          UPDATE cameras 
          SET room_id = ?
          WHERE id = ?
      """, (room_id, camera_id))
      await db.commit()
    
  async def get_camera_room(self, camera_id: int):
    async with aiosqlite.connect(self.db_path) as db:
      async with db.execute("""
          SELECT room_id FROM cameras
          WHERE id = ?
      """, (camera_id,)) as cursor:
        return await cursor.fetchone()
      
  async def get_alarms(self):
    async with aiosqlite.connect(self.db_path) as db:
      db.row_factory = aiosqlite.Row
      async with db.execute("SELECT * FROM alarm") as cursor:
        return [dict(row) for row in await cursor.fetchall()]
      
  async def get_armed_alarms(self):
    async with aiosqlite.connect(self.db_path) as db:
      db.row_factory = aiosqlite.Row
      async with db.execute("SELECT * FROM alarm WHERE active = 1") as cursor:
        return [dict(row) for row in await cursor.fetchall()]
      
  async def get_alarms_room(self, room_id: int):   
    async with aiosqlite.connect(self.db_path) as db: 
      db.row_factory = aiosqlite.Row
      async with db.execute("SELECT * FROM alarm WHERE room_id = ?", (room_id,)) as cursor: 
        return [dict(row) for row in await cursor.fetchall()]