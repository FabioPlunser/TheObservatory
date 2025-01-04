from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    ForeignKey,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.future import select
from typing import List, Optional, Dict
from datetime import datetime

Base = declarative_base()


# Models
class Room(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)

    # Relationships
    cameras = relationship("Camera", back_populates="room")
    alarms = relationship("Alarm", back_populates="room")
    alarm_devices = relationship("AlarmDevice", back_populates="room")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True)
    name = Column(String)
    room_id = Column(Integer, ForeignKey("rooms.id"))
    rtsp_url = Column(String)
    status = Column(String)
    last_seen = Column(DateTime)

    # Relationships
    room = relationship("Room", back_populates="cameras")
    alarms = relationship("Alarm", back_populates="camera")


class Alarm(Base):
    __tablename__ = "alarm"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"))
    room_id = Column(Integer, ForeignKey("rooms.id"))
    timestamp = Column(DateTime)
    active = Column(Boolean)

    # Relationships
    camera = relationship("Camera", back_populates="alarms")
    room = relationship("Room", back_populates="alarms")


class AlarmDevice(Base):
    __tablename__ = "alarm_devices"

    id = Column(String, primary_key=True)
    name = Column(String)
    room_id = Column(Integer, ForeignKey("rooms.id"))
    timestamp = Column(DateTime)

    # Relationships
    room = relationship("Room", back_populates="alarm_devices")


class Database:
    def __init__(self, db_url="sqlite+aiosqlite:///db.db"):
        self.engine = create_async_engine(db_url)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_camera(
        self, camera_id: str, name: str, rtsp_url: str, status: str
    ) -> Camera:
        async with self.async_session() as session:
            camera = Camera(
                id=camera_id,
                name=name,
                rtsp_url=rtsp_url,
                status=status,
                last_seen=datetime.now(),
            )
            session.add(camera)
            await session.commit()
            return camera

    async def delete_camera(self, camera_id: str):
        async with self.async_session() as session:
            stmt = select(Camera).where(Camera.id == camera_id)
            result = await session.execute(stmt)
            camera = result.scalar_one_or_none()
            if camera:
                await session.delete(camera)
                await session.commit()

    async def get_cameras(self) -> List[Dict]:
        async with self.async_session() as session:
            stmt = select(Camera)
            result = await session.execute(stmt)
            cameras = result.scalars().all()
            return [camera.__dict__ for camera in cameras if camera]

    async def get_camera(self, camera_id: str) -> Optional[Dict]:
        async with self.async_session() as session:
            stmt = select(Camera).where(Camera.id == camera_id)
            result = await session.execute(stmt)
            camera = result.scalar_one_or_none()
            if camera:
                return camera.__dict__
            return None

    async def update_camera_status(self, camera_id: str, status: str):
        async with self.async_session() as session:
            stmt = select(Camera).where(Camera.id == camera_id)
            result = await session.execute(stmt)
            camera = result.scalar_one_or_none()
            if camera:
                camera.status = status
                camera.last_seen = datetime.now()
                await session.commit()

    async def get_rooms(self) -> List[Dict]:
        async with self.async_session() as session:
            stmt = select(Room)
            result = await session.execute(stmt)
            rooms = result.scalars().all()
            return [room.__dict__ for room in rooms]

    async def create_room(self, room_name: str) -> Room:
        async with self.async_session() as session:
            room = Room(name=room_name)
            session.add(room)
            await session.commit()
            return room

    async def assign_camera_room(self, camera_id: str, room_id: int):
        async with self.async_session() as session:
            stmt = select(Camera).where(Camera.id == camera_id)
            result = await session.execute(stmt)
            camera = result.scalar_one_or_none()
            if camera:
                camera.room_id = room_id
                await session.commit()

    async def assign_alarm_device_room(self, alarm_device_id: str, room_id: int):
        async with self.async_session() as session:
            stmt = select(AlarmDevice).where(AlarmDevice.id == alarm_device_id)
            result = await session.execute(stmt)
            alarm_device = result.scalar_one_or_none()
            if alarm_device:
                alarm_device.room_id = room_id
                await session.commit()

    async def get_alarms(self) -> List[Dict]:
        async with self.async_session() as session:
            stmt = select(Alarm)
            result = await session.execute(stmt)
            alarms = result.scalars().all()
            return [alarm.__dict__ for alarm in alarms]

    async def get_armed_alarms(self) -> List[Dict]:
        async with self.async_session() as session:
            stmt = select(Alarm).where(Alarm.active == True)
            result = await session.execute(stmt)
            alarms = result.scalars().all()
            return [
                {
                    "id": alarm.id,
                    "camera_id": alarm.camera_id,
                    "room_id": alarm.room_id,
                    "timestamp": alarm.timestamp,
                    "active": alarm.active,
                }
                for alarm in alarms
            ]

    async def get_alarms_room(self, room_id: int) -> List[Dict]:
        async with self.async_session() as session:
            stmt = select(Alarm).where(Alarm.room_id == room_id)
            result = await session.execute(stmt)
            alarms = result.scalars().all()
            return [
                {
                    "id": alarm.id,
                    "camera_id": alarm.camera_id,
                    "room_id": alarm.room_id,
                    "timestamp": alarm.timestamp,
                    "active": alarm.active,
                }
                for alarm in alarms
            ]

    async def create_alarm_device(self, alarm_device_id: str, name: str) -> AlarmDevice:
        async with self.async_session() as session:
            alarm_device = AlarmDevice(
                id=alarm_device_id, name=name, timestamp=datetime.now()
            )
            session.add(alarm_device)
            await session.commit()
            return alarm_device

    async def delete_alarm_device(self, alarm_device_id: str):
        async with self.async_session() as session:
            stmt = select(AlarmDevice).where(AlarmDevice.id == alarm_device_id)
            result = await session.execute(stmt)
            alarm_device = result.scalar_one_or_none()
            if alarm_device:
                await session.delete(alarm_device)
                await session.commit()

    async def get_alarm_devices(self) -> List[Dict]:
        async with self.async_session() as session:
            stmt = select(AlarmDevice)
            result = await session.execute(stmt)
            devices = result.scalars().all()
            return [
                {
                    "id": device.id,
                    "name": device.name,
                    "room_id": device.room_id,
                    "timestamp": device.timestamp,
                }
                for device in devices
            ]

    async def get_alarm_device(self, alarm_device_id: str) -> Optional[Dict]:
        async with self.async_session() as session:
            stmt = select(AlarmDevice).where(AlarmDevice.id == alarm_device_id)
            result = await session.execute(stmt)
            device = result.scalar_one_or_none()
            if device:
                return {
                    "id": device.id,
                    "name": device.name,
                    "room_id": device.room_id,
                    "timestamp": device.timestamp,
                }
            return None
