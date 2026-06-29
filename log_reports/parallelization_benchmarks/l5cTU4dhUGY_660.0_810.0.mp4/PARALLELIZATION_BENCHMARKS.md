# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:51:24 UTC | l5cTU4dhUGY_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.239 | 0.658 | 52.840 | 17.146 | 21.227 | 14.389 | 3.368 |

## 2026-06-26 14:51:24 UTC | l5cTU4dhUGY_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l5cTU4dhUGY_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.239` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.658 |
| save_clips | - |
| sample_frames | 0.927 |
| caption_frames | 34.571 |
| sample_fps | 2.072 |
| detect_object_yolo | 8.625 |
| audio_scan | 16.229 |
| asr_timings | 9.797 |
| ast_timings | 26.806 |
| describe_scenes | 17.146 |
| summarize_scenes | 21.227 |
| synthesize_synopsis | 14.389 |
| make_embedding | 3.368 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.504 |
| branch_yolo_total | 10.703 |
| branch_audio_total | 52.840 |
