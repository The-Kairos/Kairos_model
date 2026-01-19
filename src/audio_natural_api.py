# src/audio_natural_api.py
import os
import time
from google.cloud import videointelligence

def extract_natural_audio_api(audio_path: str, gcloud_json_path: str, enable_logs=True):
    """
    Detect environmental / non-speech audio labels using Google Cloud Audio Intelligence API.
    
    Args:
        audio_path (str): Local path or GCS path to audio/video
        gcloud_json_path (str): Path to Service Account JSON
        enable_logs (bool): Print debug info
    
    Returns:
        labels (list of dict): [{'start_time_sec':..., 'end_time_sec':..., 'labels':[...]}]
        timings (dict)
    """
    timings = {}
    t0 = time.time()
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcloud_json_path
    client = videointelligence.VideoIntelligenceServiceClient()

    request = {
        "input_uri": audio_path,  # can also use GCS URI
        "features": ["LABEL_DETECTION"]
    }
    
    operation = client.annotate_video(request)
    result = operation.result(timeout=600)  # wait for operation to finish
    
    labels_out = []
    for annotation in result.annotation_results[0].segment_label_annotations:
        for segment in annotation.segments:
            labels_out.append({
                "start_time_sec": segment.segment.start_time_offset.seconds + segment.segment.start_time_offset.microseconds * 1e-6,
                "end_time_sec": segment.segment.end_time_offset.seconds + segment.segment.end_time_offset.microseconds * 1e-6,
                "labels": [annotation.entity.description]
            })
    
    timings["ast_duration_sec"] = time.time() - t0
    
    if enable_logs:
        print(f"[Google Cloud Audio Intelligence] {audio_path} -> {len(labels_out)} labels")
        print("AST timings:", timings)
    
    return labels_out, timings
