from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from multiprocessing import Process, Event
from app.model import detect_objects_in_frame,model
import numpy as np
import cv2
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TargetRequest(BaseModel):
    targets: list[str]

detection_process = None
stop_event = Event()

def run_model(targets):
    model(targets, stop_event=stop_event)

@app.get("/")
def serve_index():
    return FileResponse("app/index.html", media_type="text/html")

@app.post("/start_detection")
def start_detection(req: TargetRequest):
    global detection_process, stop_event

    if detection_process and detection_process.is_alive():
        raise HTTPException(status_code=400, detail="La detección ya está en ejecución.")

    stop_event.clear()
    detection_process = Process(target=run_model, args=(req.targets,))
    detection_process.start()
    return {"message": "Detección iniciada"}

@app.post("/stop_detection")
def stop_detection():
    global detection_process, stop_event

    if not detection_process or not detection_process.is_alive():
        raise HTTPException(status_code=400, detail="No hay una detección en ejecución.")

    stop_event.set()
    detection_process.join()
    detection_process = None
    return {"message": "Detección detenida"}

@app.post("/detect_frame")
async def detect_frame(frame: UploadFile = File(...), targets: str = Form(...)):
    contents = await frame.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        selected_targets = json.loads(targets)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato de 'targets' inválido")
    
    detections = detect_objects_in_frame(img, selected_targets)
    return {"detections": detections}
