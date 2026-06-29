# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:35:35 UTC | uh2qGWfmESk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.114 | 0.775 | 69.912 | 15.524 | 10.059 | 12.945 | 8.604 |

## 2026-06-27 01:35:35 UTC | uh2qGWfmESk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uh2qGWfmESk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.114` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 1.451 |
| caption_frames | 60.218 |
| sample_fps | 2.568 |
| detect_object_yolo | 11.649 |
| audio_scan | 13.968 |
| asr_timings | 12.166 |
| ast_timings | 43.769 |
| describe_scenes | 15.524 |
| summarize_scenes | 10.059 |
| synthesize_synopsis | 12.945 |
| make_embedding | 8.604 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.675 |
| branch_yolo_total | 14.222 |
| branch_audio_total | 69.912 |
