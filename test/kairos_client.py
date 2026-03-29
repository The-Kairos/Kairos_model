import argparse
import time
import requests
import json
import sys

def trigger_processing(host, port, video_path, video_id, chat_id):
    url = f"http://{host}:{port}/process"
    headers = {}
    
    files = {'video': open(video_path, 'rb')}
    data = {'videoId': video_id, 'chatId': chat_id}
    
    print(f"Sending video to VM: {url}...")
    start_time = time.time()
    response = requests.post(url, headers=headers, files=files, data=data)
    elapsed = time.time() - start_time
    
    if response.status_code == 202:
        run_id = response.json().get("runId")
        print(f"Success! Job started. RunID: {run_id} (Response Time: {elapsed:.2f}s)")
        return run_id
    else:
        print(f"Error starting job: {response.status_code}")
        print(response.text)
        return None

def watch_progress(host, port, run_id):
    url = f"http://{host}:{port}/jobs/{run_id}/stream"
    headers = {}
    
    print(f"Opening SSE stream: {url}...")
    print("=" * 60)
    
    try:
        # We use stream=True to handle the SSE stream
        response = requests.get(url, headers=headers, stream=True)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_json = json.loads(decoded_line[6:])
                    stage = data_json.get("stage", "N/A")
                    status = data_json.get("status", "N/A")
                    print(f"[{time.strftime('%H:%M:%S')}] Status: {status} | Stage: {stage}")
                    
                    if status in ('completed', 'failed'):
                        print("=" * 60)
                        print(f"Processing finished with status: {status}")
                        break
    except KeyboardInterrupt:
        print("\nDisconnected from stream.")
    except Exception as e:
        print(f"Error watching stream: {e}")

def main():
    parser = argparse.ArgumentParser(description="Kairos VM Client - Trigger and watch video processing.")
    parser.add_argument("--host", default="localhost", help="VM external IP or hostname")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--video", required=True, help="Path to local video file")
    parser.add_argument("--video-id", default="test_vid_001", help="System Video ID")
    parser.add_argument("--chat-id", default="65f4d1a2b3c4d5e6f7a8b9c0", help="MongoDB Chat ID")
    
    args = parser.parse_args()
    
    run_id = trigger_processing(args.host, args.port, args.video, args.video_id, args.chat_id)
    
    if run_id:
        watch_progress(args.host, args.port, run_id)

if __name__ == "__main__":
    main()
