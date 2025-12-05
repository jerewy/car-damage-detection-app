from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import uuid
import logging
import time
from datetime import datetime
from .config import UPLOAD_DIR
from .utils import run_inference, generate_insurance_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Damage Detection System")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    make: str = Form(...),
    model: str = Form(...),
    year: str = Form(...)
):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save uploaded file
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        input_path = UPLOAD_DIR / unique_filename
        output_filename = f"processed_{unique_filename}"
        output_path = UPLOAD_DIR / output_filename
        
        logger.info(f"Saving uploaded file to {input_path}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run Inference
        logger.info("Running inference...")
        inference_result = run_inference(str(input_path), str(output_path))
        
        # Generate Report
        logger.info("Generating report...")
        report = generate_insurance_report(inference_result, make, model, year)
        
        # Generate Report ID
        report_id = f"DA-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
        
        return JSONResponse({
            "status": "success",
            "original_image": f"/static/uploads/{unique_filename}",
            "processed_image": f"/static/uploads/{output_filename}",
            "inference_data": inference_result,
            "report": report,
            "report_meta": {
                "id": report_id,
                "timestamp": datetime.now().strftime("%B %d, %Y, %I:%M %p")
            }
        })

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
