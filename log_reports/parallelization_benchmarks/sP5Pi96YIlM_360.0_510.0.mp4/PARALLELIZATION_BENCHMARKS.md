# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:49:09 UTC | sP5Pi96YIlM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.216 | 0.777 | 63.605 | 17.362 | 11.045 | 15.415 | 4.775 |

## 2026-06-26 19:49:09 UTC | sP5Pi96YIlM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sP5Pi96YIlM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.216` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 1.270 |
| caption_frames | 51.000 |
| sample_fps | 2.296 |
| detect_object_yolo | 10.238 |
| audio_scan | 14.978 |
| asr_timings | 10.314 |
| ast_timings | 38.304 |
| describe_scenes | 17.362 |
| summarize_scenes | 11.045 |
| synthesize_synopsis | 15.415 |
| make_embedding | 4.775 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.276 |
| branch_yolo_total | 12.540 |
| branch_audio_total | 63.605 |
