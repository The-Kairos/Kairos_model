"""Regression checks for bounded YOLO tracking and scene boundaries."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.frame_obj_d_yolo import run_yolo_track_on_frames, _track_with_backoff
import src.frame_obj_d_yolo as yolo


class YoloStreamingTest(unittest.TestCase):
    def test_long_scene_is_lazy_and_submits_bounded_batches(self):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(130)]
        tracker = Mock()
        model = SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]),
                                track=Mock(side_effect=lambda chunk, **kwargs: iter([id(f) for f in chunk])))
        results = run_yolo_track_on_frames(model, frames, device='cpu')
        tracker.reset.assert_called_once()
        model.track.assert_not_called()
        self.assertEqual(next(results), id(frames[0]))
        self.assertEqual(model.track.call_count, 1)
        self.assertEqual(list(results), [id(f) for f in frames[1:]])
        self.assertEqual(model.track.call_count, 33)
        for call in model.track.call_args_list:
            self.assertLessEqual(len(call.args[0]), 4)
            self.assertTrue(call.kwargs['persist'])
            self.assertFalse(call.kwargs['half'])
            self.assertEqual(call.kwargs['imgsz'], 640)

    def test_oom_discards_partial_scene_and_resets_tracker(self):
        frames = [np.zeros((16,16,3), dtype=np.uint8) for _ in range(4)]
        tracker = Mock()
        def track(chunk, **kwargs):
            if len(chunk) == 4:
                yield 'discard this partial result'
                raise yolo.torch.cuda.OutOfMemoryError('simulated')
            yield from [id(f) for f in chunk]
        model = SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]), track=Mock(side_effect=track))
        with patch.object(yolo, '_parse_yolo_results', side_effect=lambda results, model: dict(enumerate(results))):
            results, metadata = _track_with_backoff(model, frames, 4)
        self.assertEqual(list(results.values()), [id(f) for f in frames])
        self.assertEqual(metadata['attempted_batches'], [4, 2])
        self.assertEqual(tracker.reset.call_count, 2)

    def test_oom_retries_four_two_one_and_then_fails(self):
        model = SimpleNamespace(predictor=SimpleNamespace(trackers=[]))
        with patch.object(yolo, 'run_yolo_track_on_frames', side_effect=yolo.torch.cuda.OutOfMemoryError('simulated')) as track:
            with self.assertRaises(yolo.torch.cuda.OutOfMemoryError):
                _track_with_backoff(model, [object()] * 4, 4)
        self.assertEqual([c.kwargs['batch_size'] for c in track.call_args_list], [4,2,1])

    def test_fresh_model_without_tracker_and_new_scene_reset(self):
        model = SimpleNamespace(predictor=None, track=Mock(return_value=iter([])))
        self.assertEqual(list(run_yolo_track_on_frames(model, [])), [])
        tracker = Mock()
        model.predictor = SimpleNamespace(trackers=[tracker])
        list(run_yolo_track_on_frames(model, []))
        list(run_yolo_track_on_frames(model, []))
        self.assertEqual(tracker.reset.call_count, 2)


if __name__ == '__main__':
    unittest.main()
