# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:57:47 UTC | iUCKnsDzLIs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 250.031 | 0.789 | 70.424 | 29.770 | 27.821 | 32.043 | 5.700 |

## 2026-06-26 08:57:47 UTC | iUCKnsDzLIs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iUCKnsDzLIs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `250.031` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.482 |
| caption_frames | 66.411 |
| sample_fps | 2.574 |
| detect_object_yolo | 11.604 |
| audio_scan | 12.950 |
| asr_timings | 11.318 |
| ast_timings | 46.147 |
| describe_scenes | 29.770 |
| summarize_scenes | 27.821 |
| synthesize_synopsis | 32.043 |
| make_embedding | 5.700 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 67.900 |
| branch_yolo_total | 14.184 |
| branch_audio_total | 70.424 |
