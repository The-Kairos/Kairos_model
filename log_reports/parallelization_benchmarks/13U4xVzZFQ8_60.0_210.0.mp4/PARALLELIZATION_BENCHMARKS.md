# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:14:46 UTC | 13U4xVzZFQ8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.945 | 1.943 | 63.968 | 12.301 | 8.378 | 6.261 | 5.079 |
| 2026-06-21 20:54:10 UTC | 13U4xVzZFQ8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 13:25:53 UTC | 13U4xVzZFQ8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 256.903 | 1.966 | 64.256 | 25.986 | 30.843 | 55.451 | 4.987 |

## 2026-06-21 09:14:46 UTC | 13U4xVzZFQ8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13U4xVzZFQ8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.945` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.943 |
| save_clips | - |
| sample_frames | 3.763 |
| caption_frames | 51.013 |
| sample_fps | 6.593 |
| detect_object_yolo | 10.338 |
| audio_scan | 14.757 |
| asr_timings | 8.726 |
| ast_timings | 40.477 |
| describe_scenes | 12.301 |
| summarize_scenes | 8.378 |
| synthesize_synopsis | 6.261 |
| make_embedding | 5.079 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.782 |
| branch_yolo_total | 16.937 |
| branch_audio_total | 63.968 |

## 2026-06-21 20:54:10 UTC | 13U4xVzZFQ8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13U4xVzZFQ8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.060` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 13:25:53 UTC | 13U4xVzZFQ8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13U4xVzZFQ8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `256.903` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.966 |
| save_clips | - |
| sample_frames | 3.792 |
| caption_frames | 50.852 |
| sample_fps | 6.668 |
| detect_object_yolo | 10.708 |
| audio_scan | 14.879 |
| asr_timings | 8.531 |
| ast_timings | 40.837 |
| describe_scenes | 25.986 |
| summarize_scenes | 30.843 |
| synthesize_synopsis | 55.451 |
| make_embedding | 4.987 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.650 |
| branch_yolo_total | 17.382 |
| branch_audio_total | 64.256 |
