# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:49:02 UTC | Kd6pVAb_tHs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.751 | 0.818 | 38.740 | 18.952 | 19.905 | 27.887 | 3.359 |

## 2026-06-25 06:49:02 UTC | Kd6pVAb_tHs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Kd6pVAb_tHs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.751` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 1.429 |
| caption_frames | 37.305 |
| sample_fps | 2.316 |
| detect_object_yolo | 8.338 |
| audio_scan | 3.821 |
| asr_timings | 0.000 |
| ast_timings | 27.205 |
| describe_scenes | 18.952 |
| summarize_scenes | 19.905 |
| synthesize_synopsis | 27.887 |
| make_embedding | 3.359 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.740 |
| branch_yolo_total | 10.660 |
| branch_audio_total | 31.034 |
