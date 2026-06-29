# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:55 UTC | 0lbehz52PFU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:42:46 UTC | 0lbehz52PFU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.584 | 0.782 | 36.084 | 9.324 | 21.661 | 20.398 | 2.133 |

## 2026-06-21 20:53:55 UTC | 0lbehz52PFU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0lbehz52PFU_60.0_210.0.mp4`
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

## 2026-06-22 12:42:46 UTC | 0lbehz52PFU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0lbehz52PFU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.584` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.470 |
| caption_frames | 17.704 |
| sample_fps | 1.896 |
| detect_object_yolo | 6.674 |
| audio_scan | 10.828 |
| asr_timings | 12.612 |
| ast_timings | 12.635 |
| describe_scenes | 9.324 |
| summarize_scenes | 21.661 |
| synthesize_synopsis | 20.398 |
| make_embedding | 2.133 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.180 |
| branch_yolo_total | 8.576 |
| branch_audio_total | 36.084 |
