# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:27:45 UTC | QhHvx4n03MU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.080 | 0.689 | 66.626 | 25.553 | 14.736 | 20.668 | 4.814 |

## 2026-06-25 15:27:45 UTC | QhHvx4n03MU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QhHvx4n03MU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.080` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.689 |
| save_clips | - |
| sample_frames | 1.539 |
| caption_frames | 49.842 |
| sample_fps | 2.420 |
| detect_object_yolo | 10.743 |
| audio_scan | 15.631 |
| asr_timings | 12.117 |
| ast_timings | 38.870 |
| describe_scenes | 25.553 |
| summarize_scenes | 14.736 |
| synthesize_synopsis | 20.668 |
| make_embedding | 4.814 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.387 |
| branch_yolo_total | 13.169 |
| branch_audio_total | 66.626 |
