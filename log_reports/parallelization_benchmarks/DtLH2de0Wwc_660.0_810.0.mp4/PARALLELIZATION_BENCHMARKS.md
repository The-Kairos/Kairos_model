# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:12:47 UTC | DtLH2de0Wwc_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 145.648 | 0.758 | 75.067 | 6.402 | 5.886 | 11.188 | 2.773 |

## 2026-06-24 23:12:47 UTC | DtLH2de0Wwc_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DtLH2de0Wwc_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.648` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.758 |
| save_clips | - |
| sample_frames | 0.885 |
| caption_frames | 31.229 |
| sample_fps | 2.039 |
| detect_object_yolo | 7.994 |
| audio_scan | 13.841 |
| asr_timings | 39.014 |
| ast_timings | 22.203 |
| describe_scenes | 6.402 |
| summarize_scenes | 5.886 |
| synthesize_synopsis | 11.188 |
| make_embedding | 2.773 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.120 |
| branch_yolo_total | 10.039 |
| branch_audio_total | 75.067 |
