# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:28:09 UTC | MQ1bAV_vIRI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.370 | 0.781 | 38.220 | 9.580 | 11.541 | 22.805 | 2.124 |

## 2026-06-25 09:28:09 UTC | MQ1bAV_vIRI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MQ1bAV_vIRI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.370` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 0.387 |
| caption_frames | 19.943 |
| sample_fps | 1.907 |
| detect_object_yolo | 6.636 |
| audio_scan | 13.755 |
| asr_timings | 11.693 |
| ast_timings | 12.764 |
| describe_scenes | 9.580 |
| summarize_scenes | 11.541 |
| synthesize_synopsis | 22.805 |
| make_embedding | 2.124 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.335 |
| branch_yolo_total | 8.548 |
| branch_audio_total | 38.220 |
