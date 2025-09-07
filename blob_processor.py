import numpy as np
import cv2
import tempfile
import os

class BlobTracker:
    def __init__(self):
        self.prev_centers = None
    
    def parse_color(self, color_str):
        """Parse color string 'R,G,B' to tuple"""
        try:
            return tuple(map(int, color_str.split(',')))
        except:
            return (255, 255, 255)  # Default white

    def process_image(self, image_data, params):
        """Process image data and return processed image"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Convert to grayscale for blob detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        threshold_val = params.get('threshold', 127)
        _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Set up blob detector parameters
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.minArea = params.get('min_area', 100)
        detector_params.maxArea = params.get('max_area', 10000)
        detector_params.filterByCircularity = False
        detector_params.filterByConvexity = False
        detector_params.filterByInertia = False
        detector_params.minThreshold = 1
        detector_params.maxThreshold = 255

        # Create detector and find blobs
        detector = cv2.SimpleBlobDetector_create(detector_params)
        keypoints = detector.detect(thresh)
        
        # Limit number of blobs
        max_blobs = params.get('max_blobs', 100)
        keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:max_blobs]
        
        # Draw blobs on original image
        out_img = img.copy()
        outline_color = self.parse_color(params.get('outline_color', '255,255,255'))
        blob_thickness = params.get('blob_thickness', 2)
        
        for idx, kp in enumerate(keypoints):
            cx, cy = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            
            # Calculate bounding box
            half_size = size // 2
            x0 = max(cx - half_size, 0)
            y0 = max(cy - half_size, 0)
            x1 = min(cx + half_size, img.shape[1])
            y1 = min(cy + half_size, img.shape[0])
            
            # Draw rectangle around blob
            if blob_thickness == -1:
                cv2.rectangle(out_img, (x0, y0), (x1, y1), outline_color, -1)
            else:
                cv2.rectangle(out_img, (x0, y0), (x1, y1), outline_color, blob_thickness)
            
            # Draw ID if requested
            if params.get('show_ids', False):
                text = f"ID {idx}"
                font_scale = 0.5
                cv2.putText(out_img, text, (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, outline_color, 1)
        
        # Encode result back to bytes
        _, buffer = cv2.imencode('.png', out_img)
        return buffer.tobytes()

    def process_video_frame_by_frame(self, video_data, params):
        """Process video and return processed video"""
        # Save input video to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_input:
            tmp_input.write(video_data)
            tmp_input_path = tmp_input.name
        
        # Create output path
        tmp_output_path = tempfile.mktemp(suffix='.mp4')
        
        try:
            # Open video
            cap = cv2.VideoCapture(tmp_input_path)
            if not cap.isOpened():
                raise ValueError("Could not open video")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to bytes and process
                _, frame_bytes = cv2.imencode('.png', frame)
                processed_frame_bytes = self.process_image(frame_bytes.tobytes(), params)
                
                # Convert back to frame
                nparr = np.frombuffer(processed_frame_bytes, np.uint8)
                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Write frame
                out.write(processed_frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            # Read processed video
            with open(tmp_output_path, 'rb') as f:
                result = f.read()
            
            return result, frame_count
            
        finally:
            # Cleanup
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
