# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:09:00 UTC | 13U4xVzZFQ8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.252 | 1.970 | 68.253 | 11.861 | 12.098 | 12.780 | 4.866 |
| 2026-06-21 20:54:08 UTC | 13U4xVzZFQ8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-22 13:17:40 UTC | 13U4xVzZFQ8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 249.338 | 1.996 | 66.103 | 26.130 | 48.063 | 28.046 | 4.762 |

## 2026-06-21 09:09:00 UTC | 13U4xVzZFQ8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13U4xVzZFQ8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.252` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.970 |
| save_clips | - |
| sample_frames | 3.566 |
| caption_frames | 50.469 |
| sample_fps | 6.720 |
| detect_object_yolo | 10.369 |
| audio_scan | 12.726 |
| asr_timings | 17.597 |
| ast_timings | 37.922 |
| describe_scenes | 11.861 |
| summarize_scenes | 12.098 |
| synthesize_synopsis | 12.780 |
| make_embedding | 4.866 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.041 |
| branch_yolo_total | 17.094 |
| branch_audio_total | 68.253 |

## 2026-06-21 20:54:08 UTC | 13U4xVzZFQ8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13U4xVzZFQ8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.059` sec

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

## 2026-06-22 13:17:40 UTC | 13U4xVzZFQ8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13U4xVzZFQ8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `249.338` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.996 |
| save_clips | - |
| sample_frames | 3.659 |
| caption_frames | 51.193 |
| sample_fps | 6.912 |
| detect_object_yolo | 11.024 |
| audio_scan | 12.926 |
| asr_timings | 14.748 |
| ast_timings | 38.420 |
| describe_scenes | 26.130 |
| summarize_scenes | 48.063 |
| synthesize_synopsis | 28.046 |
| make_embedding | 4.762 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.858 |
| branch_yolo_total | 17.942 |
| branch_audio_total | 66.103 |
