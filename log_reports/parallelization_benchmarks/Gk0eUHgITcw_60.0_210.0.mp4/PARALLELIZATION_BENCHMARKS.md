# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:11:11 UTC | Gk0eUHgITcw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.890 | 0.644 | 58.087 | 10.225 | 15.333 | 13.778 | 3.782 |

## 2026-06-25 02:11:11 UTC | Gk0eUHgITcw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Gk0eUHgITcw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.890` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.644 |
| save_clips | - |
| sample_frames | 0.894 |
| caption_frames | 41.412 |
| sample_fps | 2.031 |
| detect_object_yolo | 9.307 |
| audio_scan | 11.764 |
| asr_timings | 13.640 |
| ast_timings | 32.675 |
| describe_scenes | 10.225 |
| summarize_scenes | 15.333 |
| synthesize_synopsis | 13.778 |
| make_embedding | 3.782 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.312 |
| branch_yolo_total | 11.343 |
| branch_audio_total | 58.087 |
