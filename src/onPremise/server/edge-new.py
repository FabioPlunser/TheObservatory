from fastapi import FastAPI, File, UploadFile, Form, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import asyncio
import os
from typing import Dict
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs("static/frames", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class EdgeServer:
    def __init__(self):
        self.cameras: Dict[str, dict] = {}
        self.active_connections: Dict[str, set] = {}
        self.last_frame: Dict[str, bytes] = {}

    async def connect_client(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = set()
        self.active_connections[camera_id].add(websocket)

        try:
            while True:
                if camera_id in self.last_frame:
                    await websocket.send_text(self.last_frame[camera_id])
                await asyncio.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.active_connections[camera_id].remove(websocket)

edge_server = EdgeServer()

@app.post("/register")
async def register_camera(camera_data: dict):
    camera_id = camera_data["camera_id"]
    edge_server.cameras[camera_id] = camera_data
    logger.info(f"Registered camera {camera_id}")
    return {"status": "success"}

@app.post("/frame")
async def receive_frame(frame: UploadFile = File(...), camera_id: str = Form(...)):
    try:
        frame_data = await frame.read()
        # Store the frame as base64 for web viewing
        edge_server.last_frame[camera_id] = base64.b64encode(frame_data).decode('utf-8')
        
        # Save frame to disk
        save_path = f"static/frames/{camera_id}.jpg"
        with open(save_path, "wb") as f:
            f.write(frame_data)
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"status": "error", "message": str(e)}

@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await edge_server.connect_client(websocket, camera_id)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Streams</title>
        <style>
            .stream-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 20px;
                padding: 20px;
            }
            .camera-stream {
                border: 1px solid #ccc;
                padding: 10px;
                border-radius: 5px;
            }
            .stream-view {
                width: 100%;
                height: auto;
            }
            .status {
                margin: 10px 0;
                padding: 5px;
                border-radius: 3px;
            }
            .connected { background: #d4edda; }
            .disconnected { background: #f8d7da; }
        </style>
    </head>
    <body>
        <div class="stream-container" id="streams"></div>
        <script>
            function addCameraStream(cameraId) {
                const div = document.createElement('div');
                div.className = 'camera-stream';
                div.innerHTML = `
                    <h3>Camera ${cameraId}</h3>
                    <div id="status-${cameraId}" class="status disconnected">Connecting...</div>
                    <img id="stream-${cameraId}" class="stream-view" alt="Camera stream">
                `;
                document.getElementById('streams').appendChild(div);
                
                const ws = new WebSocket(`ws://${window.location.host}/ws/${cameraId}`);
                const img = document.getElementById(`stream-${cameraId}`);
                const status = document.getElementById(`status-${cameraId}`);
                
                ws.onopen = () => {
                    status.textContent = 'Connected';
                    status.className = 'status connected';
                };
                
                ws.onmessage = (event) => {
                    img.src = 'data:image/jpeg;base64,' + event.data;
                };
                
                ws.onclose = () => {
                    status.textContent = 'Disconnected';
                    status.className = 'status disconnected';
                };
            }
            
            // Fetch available cameras and set up streams
            fetch('/cameras')
                .then(response => response.json())
                .then(data => {
                    for (const camera of data.cameras) {
                        addCameraStream(camera.camera_id);
                    }
                });
        </script>
    </body>
    </html>
    """

@app.get("/cameras")
async def get_cameras():
    return {"cameras": [{"camera_id": id, **data} for id, data in edge_server.cameras.items()]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)