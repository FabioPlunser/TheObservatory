import os
import logging
import signal
import psutil
import uvicorn

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from edge_server import EdgeServer
from database import Database
from routes import Router
from nats_client import SharedNatsClient
from logging_config import setup_logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

setup_logger()
logger = logging.getLogger("Main")

PORT = 8000
edge_server = EdgeServer(PORT)
db = Database()


def kill_child_processes():
    """Kill all child processes of the current process"""
    try:
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)

        for child in children:
            try:
                child.kill()
            except:
                pass

        logger.info(f"Killed {len(children)} child processes")
    except Exception as e:
        logger.error(f"Error killing child processes: {e}")


def signal_handler(signum, frame):
    """Handle Ctrl+C by killing all child processes"""
    logger.info("Shutdown signal received, killing all processes...")
    kill_child_processes()
    os._exit(0)  # Force exit


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await db.init_db()

        cloud_url = await db.get_cloud_url()
        if cloud_url:
            try:
                nats_client = await SharedNatsClient.initialize(cloud_url)
                if not nats_client:
                    logger.error("Failed to initialize NATS client, removing cloud URL")
                    await db.update_cloud_url(None)
            except Exception as e:
                logger.error(f"Error connecting to NATS: {e}")
                logger.info("Removing cloud URL due to connection failure")
                await db.update_cloud_url(None)
        else:
            logger.info("No cloud url found in database")

        await edge_server.start_mdns()
        await edge_server.init_bucket()

        # Start cameras from database
        cameras = await db.get_cameras()
        for camera in cameras:
            try:
                if camera["rtsp_url"]:
                    await edge_server.start_camera_stream(
                        camera["id"], camera["rtsp_url"]
                    )
            except Exception as e:
                logger.error(f"Error starting camera {camera['id']}: {e}")

        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        kill_child_processes()


def create_app():
    app = FastAPI(lifespan=lifespan)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.include_router(Router(edge_server, db).create_routes())

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUILD_DIR = os.path.join(BASE_DIR, "website", "build")
    app.mount("/", StaticFiles(directory=BUILD_DIR, html=True), name="static")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
