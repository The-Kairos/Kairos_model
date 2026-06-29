# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:51:48 UTC | WfPxnxyDI7I_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 102.453 | 0.670 | 33.352 | 7.600 | 15.374 | 15.155 | 2.023 |

## 2026-06-25 20:51:48 UTC | WfPxnxyDI7I_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WfPxnxyDI7I_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `102.453` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.670 |
| save_clips | - |
| sample_frames | 0.340 |
| caption_frames | 17.980 |
| sample_fps | 1.713 |
| detect_object_yolo | 6.849 |
| audio_scan | 9.618 |
| asr_timings | 10.608 |
| ast_timings | 13.117 |
| describe_scenes | 7.600 |
| summarize_scenes | 15.374 |
| synthesize_synopsis | 15.155 |
| make_embedding | 2.023 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.326 |
| branch_yolo_total | 8.567 |
| branch_audio_total | 33.352 |
