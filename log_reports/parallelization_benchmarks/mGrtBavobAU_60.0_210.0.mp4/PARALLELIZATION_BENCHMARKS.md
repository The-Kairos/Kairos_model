# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:13:49 UTC | mGrtBavobAU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.706 | 0.824 | 37.975 | 8.684 | 8.209 | 21.899 | 2.448 |

## 2026-06-26 18:13:49 UTC | mGrtBavobAU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mGrtBavobAU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.706` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.824 |
| save_clips | - |
| sample_frames | 0.573 |
| caption_frames | 20.302 |
| sample_fps | 2.019 |
| detect_object_yolo | 6.356 |
| audio_scan | 13.893 |
| asr_timings | 8.871 |
| ast_timings | 15.196 |
| describe_scenes | 8.684 |
| summarize_scenes | 8.209 |
| synthesize_synopsis | 21.899 |
| make_embedding | 2.448 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.881 |
| branch_yolo_total | 8.381 |
| branch_audio_total | 37.975 |
