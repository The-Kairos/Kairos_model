# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:09:23 UTC | mGrtBavobAU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 224.360 | 0.821 | 68.208 | 29.445 | 18.477 | 28.340 | 5.400 |

## 2026-06-26 18:09:23 UTC | mGrtBavobAU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mGrtBavobAU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `224.360` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.821 |
| save_clips | - |
| sample_frames | 1.827 |
| caption_frames | 56.614 |
| sample_fps | 2.633 |
| detect_object_yolo | 11.166 |
| audio_scan | 13.951 |
| asr_timings | 10.980 |
| ast_timings | 43.269 |
| describe_scenes | 29.445 |
| summarize_scenes | 18.477 |
| synthesize_synopsis | 28.340 |
| make_embedding | 5.400 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.447 |
| branch_yolo_total | 13.805 |
| branch_audio_total | 68.208 |
