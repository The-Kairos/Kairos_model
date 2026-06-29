# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:16:14 UTC | TwtZ8oinQck_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.200 | 0.910 | 45.818 | 11.609 | 14.591 | 13.640 | 3.255 |

## 2026-06-25 18:16:14 UTC | TwtZ8oinQck_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TwtZ8oinQck_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.200` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.910 |
| save_clips | - |
| sample_frames | 0.832 |
| caption_frames | 35.550 |
| sample_fps | 2.171 |
| detect_object_yolo | 8.381 |
| audio_scan | 10.698 |
| asr_timings | 8.191 |
| ast_timings | 26.921 |
| describe_scenes | 11.609 |
| summarize_scenes | 14.591 |
| synthesize_synopsis | 13.640 |
| make_embedding | 3.255 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.388 |
| branch_yolo_total | 10.558 |
| branch_audio_total | 45.818 |
