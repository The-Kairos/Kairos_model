# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:08:12 UTC | yX8oV3E5VSU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 148.891 | 0.618 | 59.000 | 11.155 | 11.350 | 9.526 | 3.570 |

## 2026-06-27 05:08:12 UTC | yX8oV3E5VSU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yX8oV3E5VSU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `148.891` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.618 |
| save_clips | - |
| sample_frames | 1.005 |
| caption_frames | 39.961 |
| sample_fps | 2.055 |
| detect_object_yolo | 9.211 |
| audio_scan | 16.263 |
| asr_timings | 12.980 |
| ast_timings | 29.749 |
| describe_scenes | 11.155 |
| summarize_scenes | 11.350 |
| synthesize_synopsis | 9.526 |
| make_embedding | 3.570 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.972 |
| branch_yolo_total | 11.272 |
| branch_audio_total | 59.000 |
