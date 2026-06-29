# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:11:31 UTC | L3o-yqXnqrE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.956 | 0.707 | 38.356 | 10.694 | 31.501 | 27.129 | 2.098 |

## 2026-06-25 07:11:31 UTC | L3o-yqXnqrE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/L3o-yqXnqrE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.956` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.707 |
| save_clips | - |
| sample_frames | 0.525 |
| caption_frames | 19.593 |
| sample_fps | 1.803 |
| detect_object_yolo | 7.117 |
| audio_scan | 14.903 |
| asr_timings | 10.581 |
| ast_timings | 12.863 |
| describe_scenes | 10.694 |
| summarize_scenes | 31.501 |
| synthesize_synopsis | 27.129 |
| make_embedding | 2.098 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.124 |
| branch_yolo_total | 8.926 |
| branch_audio_total | 38.356 |
