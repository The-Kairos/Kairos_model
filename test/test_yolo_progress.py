"""Check persisted scene recovery after interruption without model/API calls."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.yolo_progress import YoloSceneProgress
from src.redo_utils import apply_redo
import main


class YoloProgressTest(unittest.TestCase):
    def test_empty_detections_are_valid_and_settings_changes_invalidate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as directory:
            root = Path(directory)
            video = root / 'video.mkv'
            video.write_bytes(b'input identity')
            scene = {'scene_index':0, 'start_seconds':0, 'end_seconds':10,
                     'yolo_detections':[], 'yolo_inference':{'batch_size':4}}
            cache = root/'pipeline/.yolo_progress'
            progress = YoloSceneProgress(cache, video, {'conf':0.8})
            progress.save(scene)
            self.assertEqual(progress.load(scene)['yolo_detections'], [])
            apply_redo({'scenes':[dict(scene)]}, root/'pipeline', ['yolo'])
            self.assertIsNone(YoloSceneProgress(cache, video, {'conf':0.8}).load(scene))
            self.assertIsNone(progress.load(dict(scene, end_seconds=11)))
            self.assertIsNone(YoloSceneProgress(cache, video, {'conf':0.9}).load(scene))
            video.write_bytes(b'changed input identity')
            self.assertIsNone(YoloSceneProgress(cache, video, {'conf':0.8}).load(scene))

    def test_interrupted_branch_resumes_only_unfinished_scenes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as directory:
            root = Path(directory)
            video = root / 'video.mkv'
            video.write_bytes(b'input identity')
            scenes = [{'scene_index':i,'start_seconds':i*10,'end_seconds':(i+1)*10} for i in range(2)]
            def sample(**kwargs):
                return [dict(s, yolo_frames=[]) for s in kwargs['scenes']], {}
            def interrupted(**kwargs):
                done = dict(kwargs['scenes'][0], yolo_detections=[], yolo_inference={'batch_size':4})
                kwargs['scene_completed'](done)
                raise RuntimeError('interrupted after scene zero')
            with patch.object(main,'sample_fps_log', side_effect=sample), patch.object(main,'detect_object_yolo_log', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    main.run_yolo_branch(str(video), str(root/'pipeline'), scenes, False, True)
            def finish(**kwargs):
                done = [dict(s,yolo_detections=[],yolo_inference={'batch_size':4}) for s in kwargs['scenes']]
                for scene in done:
                    kwargs['scene_completed'](scene)
                return done, {}
            with patch.object(main,'sample_fps_log', side_effect=sample) as sampling, patch.object(main,'detect_object_yolo_log', side_effect=finish):
                result, _ = main.run_yolo_branch(str(video), str(root/'pipeline'), scenes, False, True)
            self.assertEqual([s['scene_index'] for s in sampling.call_args.kwargs['scenes']], [1])
            self.assertEqual(len(result),2)
            self.assertTrue(all('yolo_detections' in s for s in result))


if __name__ == '__main__':
    unittest.main()
