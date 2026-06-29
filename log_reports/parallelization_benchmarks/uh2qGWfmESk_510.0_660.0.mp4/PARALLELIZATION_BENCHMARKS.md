# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:32:19 UTC | uh2qGWfmESk_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.885 | 0.785 | 65.517 | 18.919 | 15.805 | 9.329 | 4.643 |

## 2026-06-27 01:32:19 UTC | uh2qGWfmESk_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uh2qGWfmESk_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.885` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.249 |
| caption_frames | 52.238 |
| sample_fps | 2.417 |
| detect_object_yolo | 10.581 |
| audio_scan | 14.903 |
| asr_timings | 12.519 |
| ast_timings | 38.087 |
| describe_scenes | 18.919 |
| summarize_scenes | 15.805 |
| synthesize_synopsis | 9.329 |
| make_embedding | 4.643 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.493 |
| branch_yolo_total | 13.004 |
| branch_audio_total | 65.517 |
