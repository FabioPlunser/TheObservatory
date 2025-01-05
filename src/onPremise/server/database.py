from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    ForeignKey,
    DateTime,
    Float,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.future import select
from typing import List, Optional, Dict
from datetime import datetime, timedelta

import uuid

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


class KnownFace(Base):
    __tablename__ = "known_faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    s3_key = Column(String, nullable=False)
    face_id = Column(String, nullable=False)
    face_encoding = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    detections = relationship("FaceDetection", back_populates="known_face")


class FaceDetection(Base):
    __tablename__ = "face_detections"

    camera_id = Column(String, ForeignKey("cameras.id"))
    known_face_id = Column(Integer, ForeignKey("known_faces.id"), nullable=True)
    detection_time = Column(DateTime, default=datetime.now)
    confidence = Column(Float)
    last_recognition_time = Column(DateTime)  # Track when we last did cloud recognition

    # Relationships
    camera = relationship("Camera", back_populates="face_detections")
    known_face = relationship("KnownFace", back_populates="detections")


face_detections = relationship("FaceDetection", back_populates="camera")


class Company(Base):
    __tablename__ = "company"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
    )


class Database:
    def __init__(self, db_url="sqlite+aiosqlite:///db.db"):
        self.engine = create_async_engine(db_url)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_company_id(self) -> str:
        async with self.async_session() as session:
            stmt = select(Company)
            result = await session.execute(stmt)
            company = result.scalar_one_or_none()
            if company:
                return company.id
            return None

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
            return [alarm.__dict__ for alarm in alarms]

    async def get_alarms_room(self, room_id: int) -> List[Dict]:
        async with self.async_session() as session:
            stmt = select(Alarm).where(Alarm.room_id == room_id)
            result = await session.execute(stmt)
            alarms = result.scalars().all()
            return [alarm.__dict__ for alarm in alarms]

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
            return [device.__dict__ for device in devices]

    async def get_alarm_device(self, alarm_device_id: str) -> Optional[Dict]:
        async with self.async_session() as session:
            stmt = select(AlarmDevice).where(AlarmDevice.id == alarm_device_id)
            result = await session.execute(stmt)
            device = result.scalar_one_or_none()
            if device:
                return device.__dict__
            return None

    async def create_known_face(self, face_id: str, s3_key: str) -> KnownFace:
        async with self.async_session() as session:
            known_face = KnownFace(face_id=face_id, s3_key=s3_key)
            session.add(known_face)
            await session.commit()
            return known_face

    async def delete_known_face(self, known_face_id: int):
        async with self.async_session() as session:
            stmt = select(KnownFace).where(KnownFace.id == known_face_id)
            result = await session.execute(stmt)
            known_face = result.scalar_one_or_none()
            if known_face:
                await session.delete(known_face)
                await session.commit()

    async def get_known_faces(self) -> List[Dict]:
        async with self.async_session() as session:
            stmt = select(KnownFace)
            result = await session.execute(stmt)
            faces = result.scalars().all()
            return [face.__dict__ for face in faces]

    async def create_face_detection(
        self, camera_id: str, known_face_id: Optional[int], confidence: float
    ) -> FaceDetection:
        async with self.async_session() as session:
            detection = FaceDetection(
                camera_id=camera_id,
                known_face_id=known_face_id,
                confidence=confidence,
                last_recognition_time=datetime.now(),
            )

            session.add(detection)

            await session.commit()
            return detection

    async def get_recent_face_detection(
        self, camera_id: str, known_face_id: int, timeout_minutes: int = 5
    ) -> Optional[FaceDetection]:
        async with self.async_session() as session:
            cutoff_time = datetime.now - timedelta(minutes=timeout_minutes)
            stmt = (
                select(FaceDetection)
                .where(
                    FaceDetection.camera_id == camera_id,
                    FaceDetection.known_face_id == known_face_id,
                    FaceDetection.detection_time >= cutoff_time,
                )
                .order_by(FaceDetection.detection_time.desc())
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def delete_face_detection(self, detection_id: int):
        async with self.async_session() as session:
            stmt = select(FaceDetection).where(FaceDetection.id == detection_id)
            result = await session.execute(stmt)
            detection = result.scalar_one_or_none()
            if detection:
                await session.delete(detection)
                await session.commit()
