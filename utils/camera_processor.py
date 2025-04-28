import cv2
import numpy as np
import logging
from io import BytesIO

class CameraProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Camera processor initialized")
        
    def process_image_file(self, file):
        """
        Process an uploaded image file (from request.files)
        """
        try:
            # Read file to buffer
            file_bytes = file.read()
            
            # Convert to numpy array
            nparr = np.frombuffer(file_bytes, np.uint8)
            
            # Decode to image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to RGB (MediaPipe uses RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to 480p to ensure processing is lightweight
            height, width = image_rgb.shape[:2]
            if width > 640:
                scale_factor = 640 / width
                new_width = 640
                new_height = int(height * scale_factor)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            return image_rgb
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return None
