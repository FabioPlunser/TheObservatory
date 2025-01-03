from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from edge_server import EdgeServer
from database import Database
from routes import Router

import os
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_camera_statuses(edge_server, db):
    """Check cameras and update offline status"""
    while True:
        current_time = datetime.now()
        for camera_id, data in edge_server.cameras.items():
            if (current_time - data["last_seen"]).total_seconds() > 120:
                if data["status"] != "offline":
                    edge_server.cameras[camera_id]["status"] = "offline"
                    await db.update_camera_status(camera_id, "offline")
        await asyncio.sleep(60)


edge_server = EdgeServer()
db = Database()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services
    await db.init_db()

    await edge_server.start_mdns()

    # Start background tasks
    asyncio.create_task(check_camera_statuses(edge_server, db))

    yield

    # Cleanup
    await edge_server.stop_mdns()


def create_app():
    app = FastAPI(lifespan=lifespan)

    app.include_router(Router(edge_server, db).create_routes())

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUILD_DIR = os.path.join(BASE_DIR, "website", "build")
    app.mount("/", StaticFiles(directory=BUILD_DIR, html=True), name="static")

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
