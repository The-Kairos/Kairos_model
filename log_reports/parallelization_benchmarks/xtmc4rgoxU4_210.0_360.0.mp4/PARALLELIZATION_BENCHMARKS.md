# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:16:51 UTC | xtmc4rgoxU4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.049 | 0.763 | 64.096 | 15.321 | 8.335 | 13.385 | 4.680 |

## 2026-06-27 04:16:51 UTC | xtmc4rgoxU4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xtmc4rgoxU4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.049` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.763 |
| save_clips | - |
| sample_frames | 1.272 |
| caption_frames | 51.900 |
| sample_fps | 2.429 |
| detect_object_yolo | 10.458 |
| audio_scan | 15.202 |
| asr_timings | 10.348 |
| ast_timings | 38.537 |
| describe_scenes | 15.321 |
| summarize_scenes | 8.335 |
| synthesize_synopsis | 13.385 |
| make_embedding | 4.680 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.178 |
| branch_yolo_total | 12.893 |
| branch_audio_total | 64.096 |
