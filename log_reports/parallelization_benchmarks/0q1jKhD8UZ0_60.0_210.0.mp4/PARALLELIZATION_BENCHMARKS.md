# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:58 UTC | 0q1jKhD8UZ0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:52:09 UTC | 0q1jKhD8UZ0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.060 | 0.649 | 52.981 | 20.433 | 32.714 | 23.109 | 3.391 |

## 2026-06-21 20:53:58 UTC | 0q1jKhD8UZ0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0q1jKhD8UZ0_60.0_210.0.mp4`
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

## 2026-06-22 12:52:09 UTC | 0q1jKhD8UZ0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0q1jKhD8UZ0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.060` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.649 |
| save_clips | - |
| sample_frames | 0.884 |
| caption_frames | 36.926 |
| sample_fps | 2.004 |
| detect_object_yolo | 8.601 |
| audio_scan | 15.907 |
| asr_timings | 11.053 |
| ast_timings | 26.013 |
| describe_scenes | 20.433 |
| summarize_scenes | 32.714 |
| synthesize_synopsis | 23.109 |
| make_embedding | 3.391 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.816 |
| branch_yolo_total | 10.611 |
| branch_audio_total | 52.981 |
