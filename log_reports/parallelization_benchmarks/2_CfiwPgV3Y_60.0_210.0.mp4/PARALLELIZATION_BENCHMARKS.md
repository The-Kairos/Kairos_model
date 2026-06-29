# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:48:00 UTC | 2_CfiwPgV3Y_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 141.026 | 0.776 | 43.331 | 19.073 | 9.356 | 25.029 | 2.890 |
| 2026-06-24 08:49:05 UTC | 2_CfiwPgV3Y_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.081 | 0.770 | 42.944 | 18.618 | 16.182 | 24.614 | 2.867 |

## 2026-06-23 14:48:00 UTC | 2_CfiwPgV3Y_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2_CfiwPgV3Y_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `141.026` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 0.769 |
| caption_frames | 28.325 |
| sample_fps | 2.029 |
| detect_object_yolo | 8.058 |
| audio_scan | 12.764 |
| asr_timings | 9.738 |
| ast_timings | 20.820 |
| describe_scenes | 19.073 |
| summarize_scenes | 9.356 |
| synthesize_synopsis | 25.029 |
| make_embedding | 2.890 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.101 |
| branch_yolo_total | 10.092 |
| branch_audio_total | 43.331 |

## 2026-06-24 08:49:05 UTC | 2_CfiwPgV3Y_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2_CfiwPgV3Y_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.081` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.770 |
| save_clips | - |
| sample_frames | 0.769 |
| caption_frames | 24.448 |
| sample_fps | 2.029 |
| detect_object_yolo | 8.406 |
| audio_scan | 12.850 |
| asr_timings | 9.098 |
| ast_timings | 20.987 |
| describe_scenes | 18.618 |
| summarize_scenes | 16.182 |
| synthesize_synopsis | 24.614 |
| make_embedding | 2.867 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.223 |
| branch_yolo_total | 10.441 |
| branch_audio_total | 42.944 |
