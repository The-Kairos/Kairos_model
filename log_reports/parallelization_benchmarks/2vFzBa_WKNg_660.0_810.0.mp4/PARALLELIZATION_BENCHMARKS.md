# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:05:54 UTC | 2vFzBa_WKNg_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 80.487 | 3.560 | 29.725 | 3.619 | 3.925 | 7.662 | 1.918 |
| 2026-06-21 21:45:20 UTC | 2vFzBa_WKNg_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 80.831 | 3.682 | 30.151 | 3.726 | 2.666 | 6.584 | 1.825 |

## 2026-06-21 10:05:54 UTC | 2vFzBa_WKNg_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `80.487` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.560 |
| save_clips | - |
| sample_frames | 1.066 |
| caption_frames | 13.007 |
| sample_fps | 8.799 |
| detect_object_yolo | 5.896 |
| audio_scan | 6.484 |
| asr_timings | 13.332 |
| ast_timings | 9.901 |
| describe_scenes | 3.619 |
| summarize_scenes | 3.925 |
| synthesize_synopsis | 7.662 |
| make_embedding | 1.918 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.079 |
| branch_yolo_total | 14.701 |
| branch_audio_total | 29.725 |

## 2026-06-21 21:45:20 UTC | 2vFzBa_WKNg_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `80.831` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.682 |
| save_clips | - |
| sample_frames | 1.097 |
| caption_frames | 14.129 |
| sample_fps | 9.312 |
| detect_object_yolo | 6.180 |
| audio_scan | 6.628 |
| asr_timings | 13.452 |
| ast_timings | 10.062 |
| describe_scenes | 3.726 |
| summarize_scenes | 2.666 |
| synthesize_synopsis | 6.584 |
| make_embedding | 1.825 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.232 |
| branch_yolo_total | 15.498 |
| branch_audio_total | 30.151 |
