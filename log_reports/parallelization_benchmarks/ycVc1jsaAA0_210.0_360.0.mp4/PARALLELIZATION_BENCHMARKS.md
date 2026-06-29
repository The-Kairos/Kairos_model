# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:16:34 UTC | ycVc1jsaAA0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.845 | 0.830 | 60.376 | 13.193 | 10.445 | 8.356 | 4.962 |

## 2026-06-27 05:16:34 UTC | ycVc1jsaAA0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ycVc1jsaAA0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.845` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.830 |
| save_clips | - |
| sample_frames | 1.448 |
| caption_frames | 53.557 |
| sample_fps | 2.512 |
| detect_object_yolo | 10.734 |
| audio_scan | 11.983 |
| asr_timings | 7.521 |
| ast_timings | 40.864 |
| describe_scenes | 13.193 |
| summarize_scenes | 10.445 |
| synthesize_synopsis | 8.356 |
| make_embedding | 4.962 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.011 |
| branch_yolo_total | 13.252 |
| branch_audio_total | 60.376 |
