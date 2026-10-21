"""Atomic per-scene YOLO summaries, scoped to input, settings, and detector code."""
import hashlib
import json
from pathlib import Path


class YoloSceneProgress:
    def __init__(self, directory, video, settings):
        self.directory = Path(directory)
        video = Path(video).resolve()
        stat = video.stat()
        detector = Path(__file__).with_name('frame_obj_d_yolo.py')
        sampling = Path(__file__).with_name('frame_sampling.py')
        generation = self.directory / 'generation.txt'
        identity = {'video': str(video), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                    'settings': settings, 'detector_sha256': hashlib.sha256(detector.read_bytes()).hexdigest(),
                    'sampling_sha256': hashlib.sha256(sampling.read_bytes()).hexdigest(),
                    'generation': generation.read_text(encoding='utf-8') if generation.exists() else ''}
        self.signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def path(self, scene):
        return self.directory / f"scene_{int(scene['scene_index']):06d}.json"

    def key(self, scene):
        return [scene['scene_index'], scene['start_seconds'], scene['end_seconds']]

    def load(self, scene):
        try:
            data = json.loads(self.path(scene).read_text(encoding='utf-8'))
            if (data['signature'] == self.signature and data['scene'] == self.key(scene)
                    and isinstance(data['yolo_detections'], list)):
                return {k: data[k] for k in ['yolo_detections', 'yolo_inference']}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def save(self, scene):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(scene)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'signature': self.signature, 'scene': self.key(scene),
            'yolo_detections': scene['yolo_detections'], 'yolo_inference': scene['yolo_inference']}), encoding='utf-8')
        temporary.replace(path)
