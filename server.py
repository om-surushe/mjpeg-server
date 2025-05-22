import os
from flask import Flask, Response
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the last frame position
last_frame_position = 0

def generate_frames(video_path):
    global last_frame_position
    camera = cv2.VideoCapture(video_path)
    
    # Set the video position to the last frame position
    camera.set(cv2.CAP_PROP_POS_FRAMES, last_frame_position)

    while True:
        success, frame = camera.read()
        if not success:
            last_frame_position = 0
            camera = cv2.VideoCapture(video_path)
            continue

        
        # Skip frames to send every 10th frame
        if last_frame_position % 1 == 0:
            # Save the frame as output.png
            cv2.imwrite("output.png", frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            print("Sending frame: ", last_frame_position)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        last_frame_position += 1

    camera.release()

@app.route('/video_feed')
def video_feed():
    video_path = os.environ.get('VIDEO_PATH')  # Get the video path from an environment variable
    if video_path is None:
        return 'Video path not specified'
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # return json response status : working
    return {'status': 'working'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')