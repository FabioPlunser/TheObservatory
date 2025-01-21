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
from logging_config import setup_logger

import logging

setup_logger()
logger = logging.getLogger("Database")

import uuid

Base = declarative_base()


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True)
    name = Column(String)
    rtsp_url = Column(String)
    status = Column(String)
    last_seen = Column(DateTime)

    detected_unknown_face = Column(Boolean)
    uknown_face_url = Column(String)

    face_detections = relationship("FaceDetection", back_populates="camera")  # Add this


class Alarm(Base):
    __tablename__ = "alarm"

    id = Column(String, primary_key=True)
    last_seen = Column(DateTime)
    active = Column(Boolean)
    # manual_deactivation = Column(Boolean)
    last_trigger = Column(DateTime)
    connected = Column(Boolean)


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

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=True)
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
        unique=True,
        nullable=False,
    )
    name = Column(String, nullable=True)
    cloud_url = Column(String, nullable=True)
    init_bucket = Column(Boolean, default=False)


class ReidConfig(Base):
    __tablename__ = "reid_config"
    id = Column(Integer, primary_key=True, autoincrement=True)
    max_person_age = Column(Integer, default=3600)
    reid_threshold = Column(Float, default=0.5)
    spatial_weight = Column(Float, default=0.5)
    temporal_weight = Column(Float, default=0.5)


class PersonTracking(Base):
    __tablename__ = "person_tracking"

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(String, nullable=False)
    last_seen = Column(DateTime, nullable=False)
    face_id = Column(String, nullable=True)
    recognition_status = Column(String, nullable=False, default="pending")
    camera_ids = Column(JSON, nullable=False)  # List of cameras where person was seen
    face_embedding = Column(
        JSON, nullable=True
    )  # Store face embedding for faster matching


class TrackingConfig(Base):
    __tablename__ = "tracking_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stale_track_timeout = Column(Integer, default=1800)  # 30 minutes in seconds
    reid_timeout = Column(Integer, default=3600)  # 1 hour in seconds
    face_recognition_timeout = Column(Integer, default=7200)  # 2 hours in seconds
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Database:
    def __init__(self, db_url="sqlite+aiosqlite:///db.db"):
        self.engine = create_async_engine(db_url)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        company = await self.get_company()
        if not company:
            await self.create_company(None, None)

    async def create_company(self, name: str, cloud_url: str):
        async with self.async_session() as session:
            company_id = str(uuid.uuid4())
            company = Company(
                id=company_id,
                name=name,
                cloud_url=cloud_url,
            )
            session.add(company)
            await session.commit()
            return company_id

    async def get_company_id(self) -> str:
        async with self.async_session() as session:
            stmt = select(Company)
            result = await session.execute(stmt)
            company = result.scalar_one_or_none()
            if company:
                return company.id
            return None

    async def get_company_init_bucket(self) -> bool:
        async with self.async_session() as session:
            stmt = select(Company)
            result = await session.execute(stmt)
            company = result.scalar_one_or_none()
            if company:
                return company.init_bucket
            return None

    async def update_company_init_bucket(self, init_bucket: bool):
        async with self.async_session() as session:
            stmt = select(Company)
            result = await session.execute(stmt)
            company = result.scalar_one_or_none()
            if company:
                company.init_bucket = init_bucket
                await session.commit()

    async def update_cloud_url(self, cloud_url: str):
        async with self.async_session() as session:
            stmt = select(Company)
            result = await session.execute(stmt)
            company = result.scalar_one_or_none()
            if company:
                company.cloud_url = cloud_url
                await session.commit()

    async def get_cloud_url(self) -> str:
        async with self.async_session() as session:
            stmt = select(Company)
            result = await session.execute(stmt)
            company = result.scalar_one_or_none()
            if company:
                return company.cloud_url
            return None

    async def update_company(self, company_id: str, cloud_url: str, cloud_api_key: str):
        async with self.async_session() as session:
            stmt = select(Company).where(Company.id == company_id)
            result = await session.execute(stmt)
            company = result.scalar_one_or_none()
            if company:
                company.cloud_url = cloud_url
                company.cloud_api_key = cloud_api_key
                await session.commit()

    async def get_company(self) -> Dict:
        async with self.async_session() as session:
            stmt = select(Company)
            result = await session.execute(stmt)
            company = result.scalar_one_or_none()
            if company:
                return company.__dict__
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

    async def update_camera_unknown_face(
        self, camera_id: str, detected_unknown_face: bool, unknown_face_url: str
    ):
        async with self.async_session() as session:
            stmt = select(Camera).where(Camera.id == camera_id)
            result = await session.execute(stmt)
            camera = result.scalar_one_or_none()
            if camera:
                camera.detected_unknown_face = detected_unknown_face
                camera.unknown_face_url = unknown_face_url
                await session.commit()

    async def get_cameras_which_detected_unknown_face(self) -> Optional[Dict]:
        async with self.async_session() as session:
            stmt = select(Camera).where(Camera.detected_unknown_face == True)
            result = await session.execute(stmt)
            cameras = result.scalars().all()
            return [camera.__dict__ for camera in cameras]

    async def create_alarm(
        self,
        alarm_id: str,
    ) -> Alarm:
        async with self.async_session() as session:
            alarm = Alarm(
                id=alarm_id,
                connected=True,
                last_seen=datetime.now(),
                active=False,
            )
            session.add(alarm)
            await session.commit()
            return alarm

    async def delete_alarm(self, alarm_id: int):
        async with self.async_session() as session:
            stmt = select(Alarm).where(Alarm.id == alarm_id)
            result = await session.execute(stmt)
            alarm = result.scalar_one_or_none()
            if alarm:
                await session.delete(alarm)
                await session.commit()

    async def update_alarm_status(
        self,
        alarm_id: str,
        active: Optional[bool] = None,
        connected: Optional[bool] = None,
        last_trigger: Optional[datetime] = None,
    ):
        """Update alarm status with face URL"""
        async with self.async_session() as session:
            stmt = select(Alarm).where(Alarm.id == alarm_id)
            result = await session.execute(stmt)
            alarm = result.scalar_one_or_none()

            if alarm:
                if connected is not None:
                    alarm.connected = connected
                if active is not None:
                    alarm.active = active
                if last_trigger:
                    alarm.last_trigger = last_trigger
                await session.commit()

    async def get_active_alarms(self) -> List[Dict]:
        """Get currently triggered alarms"""
        async with self.async_session() as session:
            stmt = select(Alarm).where(Alarm.active == True)
            result = await session.execute(stmt)
            alarms = result.scalars().all()
            return [alarm.__dict__ for alarm in alarms]

    async def get_alarm(self, alarm_id: int) -> Optional[Dict]:
        async with self.async_session() as session:
            stmt = select(Alarm).where(Alarm.id == alarm_id)
            result = await session.execute(stmt)
            alarm = result.scalar_one_or_none()
            if alarm:
                return alarm.__dict__
            return None

    async def get_all_alarms(self) -> List[Dict]:
        async with self.async_session() as session:
            stmt = select(Alarm)
            result = await session.execute(stmt)
            alarms = result.scalars().all()
            return [alarm.__dict__ for alarm in alarms]

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

    async def get_reid_config(self) -> Dict:
        async with self.async_session() as session:
            stmt = select(ReidConfig)
            result = await session.execute(stmt)
            config = result.scalar_one_or_none()
            if not config:
                config = ReidConfig()
                session.add(config)
                await session.commit()
            return {
                "max_person_age": config.max_person_age,
                "reid_threshold": config.reid_threshold,
                "spatial_weight": config.spatial_weight,
                "temporal_weight": config.temporal_weight,
            }

    async def update_reid_config(self, **kwargs):
        async with self.async_session() as session:
            stmt = select(ReidConfig)
            result = await session.execute(stmt)
            config = result.scalar_one_or_none()
            if not config:
                config = ReidConfig()
                session.add(config)

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            await session.commit()

    async def cleanup_old_tracks(self):
        """Remove tracks that haven't been seen for longer than max_person_age"""
        async with self.async_session() as session:
            config = await self.get_reid_config()
            max_age = timedelta(seconds=config["max_person_age"])
            cutoff_time = datetime.now() - max_age

            stmt = select(PersonTracking).where(PersonTracking.last_seen < cutoff_time)
            result = await session.execute(stmt)
            old_tracks = result.scalars().all()

            for track in old_tracks:
                await session.delete(track)

            await session.commit()
            return len(old_tracks)

    async def update_person_track(
        self,
        global_id: str,
        camera_id: str,
        face_id: Optional[str] = None,
        recognition_status: Optional[str] = None,
    ):
        async with self.async_session() as session:
            stmt = select(PersonTracking).where(PersonTracking.global_id == global_id)
            result = await session.execute(stmt)
            track = result.scalar_one_or_none()

            if not track:
                track = PersonTracking(
                    global_id=global_id,
                    camera_ids=[camera_id],
                    last_seen=datetime.now(),
                )
                session.add(track)
            else:
                track.last_seen = datetime.now()
                if camera_id not in track.camera_ids:
                    track.camera_ids.append(camera_id)

            if face_id:
                track.face_id = face_id
            if recognition_status:
                track.recognition_status = recognition_status

            await session.commit()

    async def get_tracking_config(self) -> Dict:
        """Get tracking configuration"""
        async with self.async_session() as session:
            stmt = select(TrackingConfig)
            result = await session.execute(stmt)
            config = result.scalar_one_or_none()

            if not config:
                config = TrackingConfig()
                session.add(config)
                await session.commit()

            return {
                "stale_track_timeout": config.stale_track_timeout,
                "reid_timeout": config.reid_timeout,
                "face_recognition_timeout": config.face_recognition_timeout,
            }

    async def update_tracking_config(self, **kwargs):
        """Update tracking configuration"""
        async with self.async_session() as session:
            stmt = select(TrackingConfig)
            result = await session.execute(stmt)
            config = result.scalar_one_or_none()

            if not config:
                config = TrackingConfig()
                session.add(config)

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            await session.commit()
