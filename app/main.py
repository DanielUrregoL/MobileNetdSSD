from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from multiprocessing import Process, Event
from app.model import model

app = FastAPI()

# CORS para desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para recibir objetivos
class TargetRequest(BaseModel):
    targets: list[str]

# Control del proceso
detection_process = None
stop_event = Event()

# Lógica para correr el modelo en otro proceso
def run_model(targets):
    model(targets, stop_event=stop_event)

# Servir archivo HTML en "/"
@app.get("/")
def serve_index():
    return FileResponse("app/index.html", media_type="text/html")

# Endpoints de API
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


