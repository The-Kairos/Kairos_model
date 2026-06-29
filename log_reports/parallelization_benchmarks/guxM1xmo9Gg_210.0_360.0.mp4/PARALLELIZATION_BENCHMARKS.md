# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:50:06 UTC | guxM1xmo9Gg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.483 | 0.796 | 47.110 | 12.179 | 16.175 | 19.547 | 3.335 |

## 2026-06-26 05:50:06 UTC | guxM1xmo9Gg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/guxM1xmo9Gg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.483` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 0.827 |
| caption_frames | 37.359 |
| sample_fps | 2.178 |
| detect_object_yolo | 8.551 |
| audio_scan | 9.661 |
| asr_timings | 10.133 |
| ast_timings | 27.307 |
| describe_scenes | 12.179 |
| summarize_scenes | 16.175 |
| synthesize_synopsis | 19.547 |
| make_embedding | 3.335 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.192 |
| branch_yolo_total | 10.735 |
| branch_audio_total | 47.110 |
