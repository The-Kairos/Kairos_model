# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:26:34 UTC | jg80Yw8AuU0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 196.342 | 0.702 | 57.253 | 17.285 | 38.782 | 18.432 | 4.151 |

## 2026-06-26 11:26:34 UTC | jg80Yw8AuU0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jg80Yw8AuU0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `196.342` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.702 |
| save_clips | - |
| sample_frames | 1.274 |
| caption_frames | 44.784 |
| sample_fps | 2.256 |
| detect_object_yolo | 9.923 |
| audio_scan | 15.193 |
| asr_timings | 8.774 |
| ast_timings | 33.277 |
| describe_scenes | 17.285 |
| summarize_scenes | 38.782 |
| synthesize_synopsis | 18.432 |
| make_embedding | 4.151 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.064 |
| branch_yolo_total | 12.185 |
| branch_audio_total | 57.253 |
