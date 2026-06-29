# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:52:49 UTC | guxM1xmo9Gg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.903 | 0.803 | 51.992 | 16.690 | 13.375 | 19.949 | 3.663 |

## 2026-06-26 05:52:49 UTC | guxM1xmo9Gg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/guxM1xmo9Gg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.903` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 1.066 |
| caption_frames | 41.754 |
| sample_fps | 2.251 |
| detect_object_yolo | 8.952 |
| audio_scan | 12.844 |
| asr_timings | 8.680 |
| ast_timings | 30.459 |
| describe_scenes | 16.690 |
| summarize_scenes | 13.375 |
| synthesize_synopsis | 19.949 |
| make_embedding | 3.663 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.826 |
| branch_yolo_total | 11.208 |
| branch_audio_total | 51.992 |
