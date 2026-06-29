# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:42:29 UTC | l_VtY7btNRA_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.830 | 0.799 | 71.060 | 20.253 | 27.581 | 13.144 | 3.584 |

## 2026-06-26 15:42:29 UTC | l_VtY7btNRA_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l_VtY7btNRA_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.830` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 0.986 |
| caption_frames | 38.372 |
| sample_fps | 2.285 |
| detect_object_yolo | 9.286 |
| audio_scan | 14.149 |
| asr_timings | 26.690 |
| ast_timings | 30.213 |
| describe_scenes | 20.253 |
| summarize_scenes | 27.581 |
| synthesize_synopsis | 13.144 |
| make_embedding | 3.584 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.363 |
| branch_yolo_total | 11.578 |
| branch_audio_total | 71.060 |
