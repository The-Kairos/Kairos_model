# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:27:11 UTC | gehzWEPLjcc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.160 | 0.712 | 57.727 | 13.436 | 29.706 | 21.896 | 4.213 |

## 2026-06-26 05:27:11 UTC | gehzWEPLjcc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gehzWEPLjcc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.160` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.712 |
| save_clips | - |
| sample_frames | 1.922 |
| caption_frames | 47.264 |
| sample_fps | 2.471 |
| detect_object_yolo | 10.313 |
| audio_scan | 12.028 |
| asr_timings | 9.715 |
| ast_timings | 35.975 |
| describe_scenes | 13.436 |
| summarize_scenes | 29.706 |
| synthesize_synopsis | 21.896 |
| make_embedding | 4.213 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.192 |
| branch_yolo_total | 12.790 |
| branch_audio_total | 57.727 |
