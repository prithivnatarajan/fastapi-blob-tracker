from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os

app = FastAPI(title="Blob Tracker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize blob tracker (will be imported after OpenCV is installed)
blob_tracker = None

def get_blob_tracker():
    global blob_tracker
    if blob_tracker is None:
        try:
            from blob_processor import BlobTracker
            blob_tracker = BlobTracker()
        except ImportError:
            return None
    return blob_tracker

@app.get("/")
async def root():
    return {"message": "Blob Tracker API is running!", "status": "success"}

@app.get("/health")
async def health_check():
    tracker = get_blob_tracker()
    opencv_status = "available" if tracker else "not installed"
    return {
        "status": "healthy", 
        "version": "1.0.0",
        "opencv": opencv_status
    }

@app.post("/test-upload/")
async def test_upload(file: UploadFile = File(...)):
    content = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
        "status": "uploaded successfully"
    }

@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...),
    outline_color: str = Form("255,255,255"),
    blob_thickness: int = Form(2),
    min_area: float = Form(100),
    max_area: float = Form(10000),
    show_ids: bool = Form(False),
    threshold: int = Form(127)
):
    """Process uploaded image with blob detection"""
    
    # Check if OpenCV is available
    tracker = get_blob_tracker()
    if not tracker:
        raise HTTPException(status_code=500, detail="OpenCV not available")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file content
        content = await file.read()
        
        # Prepare processing parameters
        params = {
            'outline_color': outline_color,
            'blob_thickness': blob_thickness,
            'min_area': min_area,
            'max_area': max_area,
            'show_ids': show_ids,
            'threshold': threshold
        }
        
        # Process image
        processed_image = tracker.process_image(content, params)
        
        # Return processed image
        return Response(
            content=processed_image,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=processed_image.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process-video/")
async def process_video(
    file: UploadFile = File(...),
    outline_color: str = Form("255,255,255"),
    blob_thickness: int = Form(2),
    min_area: float = Form(100),
    max_area: float = Form(10000),
    show_ids: bool = Form(False),
    threshold: int = Form(127)
):
    """Process uploaded video with blob detection"""
    
    # Check if OpenCV is available
    tracker = get_blob_tracker()
    if not tracker:
        raise HTTPException(status_code=500, detail="OpenCV not available")
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Read file content
        content = await file.read()
        
        # Prepare processing parameters
        params = {
            'outline_color': outline_color,
            'blob_thickness': blob_thickness,
            'min_area': min_area,
            'max_area': max_area,
            'show_ids': show_ids,
            'threshold': threshold
        }
        
        # Process video
        processed_video, frame_count = tracker.process_video_frame_by_frame(content, params)
        
        # Return processed video
        return Response(
            content=processed_video,
            media_type="video/mp4",
            headers={
                "Content-Disposition": "attachment; filename=processed_video.mp4",
                "X-Frame-Count": str(frame_count)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
