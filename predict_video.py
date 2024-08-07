import os
import cv2
import argparse

from ultralytics import YOLO

def process_video(video_path):
    #get the video
    VIDEOS_DIR = os.path.join('.', 'videos')
    #if empty
    if video_path == '': 
        video_path = os.path.join(VIDEOS_DIR, 'traffic road.mp4')#default path
    else:
        video_path = os.path.join(VIDEOS_DIR, video_path)

    #path of the result video
    video_path_out = '{}_out.mp4'.format(video_path)

    #get frames of the videos
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    # Load a model
    model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')
    model = YOLO(model_path)  # load a custom model

    threshold = 0.7

    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            #filter the result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument('--path', type=str, required=True, help='relative path to the video file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with the provided path
    process_video(args.path)
