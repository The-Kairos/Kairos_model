# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:50:55 UTC | 8e8VeSBgq6k_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.496 | 0.800 | 69.420 | 14.820 | 15.006 | 17.527 | 5.317 |

## 2026-06-24 16:50:55 UTC | 8e8VeSBgq6k_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8e8VeSBgq6k_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.496` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 1.237 |
| caption_frames | 55.366 |
| sample_fps | 2.394 |
| detect_object_yolo | 11.202 |
| audio_scan | 14.916 |
| asr_timings | 11.653 |
| ast_timings | 42.843 |
| describe_scenes | 14.820 |
| summarize_scenes | 15.006 |
| synthesize_synopsis | 17.527 |
| make_embedding | 5.317 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.609 |
| branch_yolo_total | 13.602 |
| branch_audio_total | 69.420 |
