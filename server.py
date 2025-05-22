import os
from flask import Flask, Response, abort
import cv2
from flask_cors import CORS
import threading
import time
from contextlib import contextmanager
import logging
from threading import Lock, Event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Thread-safe state management
frame_lock = Lock()
connection_lock = Lock()
stream_state = {
    'frame_position': 0,
    'active_connections': 0,
    'is_streaming': Event()
}

# Configuration from environment variables with validation
def get_env_int(key, default, min_val, max_val=None):
    """Get and validate integer environment variables."""
    value = int(os.environ.get(key, default))
    if value < min_val or (max_val and value > max_val):
        raise ValueError(f"{key} must be between {min_val} and {max_val}")
    return value

FRAME_QUALITY = get_env_int('FRAME_QUALITY', 80, 1, 100)
MAX_FPS = get_env_int('MAX_FPS', 30, 1, 60)
FRAME_BUFFER_SIZE = get_env_int('FRAME_BUFFER_SIZE', 10, 1, 100)
FRAME_SKIP = get_env_int('FRAME_SKIP', 1, 1, 10)  # New: Allow frame skipping for performance

class FrameBuffer:
    """Thread-safe frame buffer with size limit and error handling."""
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.lock = Lock()
    
    def add_frame(self, frame):
        """Add a frame to the buffer with thread safety."""
        if frame is None:
            raise ValueError("Cannot add None frame to buffer")
            
        with self.lock:
            self.buffer.append(frame)
            while len(self.buffer) > self.max_size:
                self.buffer.pop(0)
    
    def get_latest_frame(self):
        """Get the most recent frame with thread safety."""
        with self.lock:
            return self.buffer[-1] if self.buffer else None
    
    def clear(self):
        """Clear the buffer with thread safety."""
        with self.lock:
            self.buffer.clear()

frame_buffer = FrameBuffer(FRAME_BUFFER_SIZE)

@contextmanager
def video_capture(path):
    """Context manager for handling video capture resources."""
    camera = None
    try:
        camera = cv2.VideoCapture(path)
        if not camera.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        yield camera
    finally:
        if camera is not None:
            camera.release()
            logger.info("Video capture released")

@contextmanager
def track_connection():
    """
    Enhanced context manager for tracking active connections and stream lifecycle.
    Properly handles connection counting and stream state management.
    """
    try:
        with connection_lock:
            stream_state['active_connections'] += 1
            if stream_state['active_connections'] == 1:
                # First connection triggers streaming
                stream_state['is_streaming'].set()
            logger.info(f"New connection. Total active: {stream_state['active_connections']}")
        
        yield
        
    finally:
        with connection_lock:
            stream_state['active_connections'] = max(0, stream_state['active_connections'] - 1)
            if stream_state['active_connections'] == 0:
                # Last connection cleanup
                stream_state['is_streaming'].clear()
                frame_buffer.clear()
                logger.info("Last connection closed, cleared frame buffer")
            logger.info(f"Connection closed. Total active: {stream_state['active_connections']}")

def generate_frames(video_path):
    """Generate video frames with improved resource management and performance."""
    frame_time = 1.0 / MAX_FPS
    last_frame_time = 0
    
    with track_connection():
        try:
            with video_capture(video_path) as camera:
                while stream_state['is_streaming'].is_set():
                    # Frame rate limiting
                    current_time = time.time()
                    time_elapsed = current_time - last_frame_time
                    if time_elapsed < frame_time:
                        time.sleep(frame_time - time_elapsed)
                    
                    with frame_lock:
                        camera.set(cv2.CAP_PROP_POS_FRAMES, stream_state['frame_position'])
                        success, frame = camera.read()
                        
                        if not success:
                            stream_state['frame_position'] = 0
                            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            logger.info("Video loop completed, restarting from beginning")
                            continue

                        try:
                            # Frame processing with quality control
                            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
                            if not ret:
                                logger.error("Failed to encode frame")
                                abort(500, description="Frame encoding failed")
                                
                            frame_data = buffer.tobytes()
                            stream_state['frame_position'] += FRAME_SKIP
                            
                            # Update frame buffer
                            frame_buffer.add_frame(frame_data)
                            
                            if stream_state['frame_position'] % 100 == 0:
                                logger.info(f"Processed frame: {stream_state['frame_position']}, "
                                          f"Active connections: {stream_state['active_connections']}")
                                
                        except cv2.error as e:
                            logger.error(f"OpenCV error during frame encoding: {e}")
                            abort(500, description="Frame processing error")
                        except Exception as e:
                            logger.error(f"Unexpected error during frame processing: {e}")
                            abort(500, description="Internal processing error")

                    last_frame_time = time.time()
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            abort(503, description="Stream generation failed")

@app.route('/video_feed')
def video_feed():
    """Stream video frames with enhanced error handling."""
    video_path = os.environ.get('VIDEO_PATH')
    if not video_path:
        logger.error("Video path not specified in environment variables")
        abort(400, description="VIDEO_PATH environment variable not set")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        abort(404, description="Video file not found")
    
    try:
        return Response(
            generate_frames(video_path),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Error setting up video feed: {e}")
        abort(503, description="Failed to setup video feed")

@app.route('/stats')
def get_stats():
    """Get current server statistics with enhanced metrics."""
    with connection_lock:
        return {
            'active_connections': stream_state['active_connections'],
            'current_frame': stream_state['frame_position'],
            'frame_quality': FRAME_QUALITY,
            'max_fps': MAX_FPS,
            'buffer_size': FRAME_BUFFER_SIZE,
            'frame_skip': FRAME_SKIP,
            'is_streaming': stream_state['is_streaming'].is_set()
        }

@app.route('/')
def index():
    """Health check endpoint."""
    return {'status': 'working', 'version': '1.2.0'}

# Error handlers
@app.errorhandler(400)
def bad_request(e):
    """Handle client errors."""
    return {"error": str(e.description)}, 400

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors."""
    return {"error": str(e.description)}, 404

@app.errorhandler(500)
def server_error(e):
    """Handle server errors."""
    return {"error": str(e.description)}, 500

@app.errorhandler(503)
def service_unavailable(e):
    """Handle service unavailable errors."""
    return {"error": str(e.description)}, 503

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)