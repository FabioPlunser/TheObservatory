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
import signal
import psutil  # Make sure to pip install psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        await edge_server.start_mdns()

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

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
