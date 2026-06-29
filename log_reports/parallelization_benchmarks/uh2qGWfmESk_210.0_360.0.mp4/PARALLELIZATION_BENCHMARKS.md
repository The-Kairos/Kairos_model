# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:26:39 UTC | uh2qGWfmESk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.721 | 0.782 | 63.663 | 14.506 | 7.367 | 7.979 | 4.644 |

## 2026-06-27 01:26:39 UTC | uh2qGWfmESk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uh2qGWfmESk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.721` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.267 |
| caption_frames | 52.949 |
| sample_fps | 2.447 |
| detect_object_yolo | 10.706 |
| audio_scan | 15.035 |
| asr_timings | 10.587 |
| ast_timings | 38.032 |
| describe_scenes | 14.506 |
| summarize_scenes | 7.367 |
| synthesize_synopsis | 7.979 |
| make_embedding | 4.644 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.222 |
| branch_yolo_total | 13.159 |
| branch_audio_total | 63.663 |
