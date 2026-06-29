# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:50:01 UTC | 5JpHeR2YWIQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.876 | 0.780 | 40.404 | 6.899 | 11.058 | 36.192 | 2.548 |

## 2026-06-24 11:50:01 UTC | 5JpHeR2YWIQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5JpHeR2YWIQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.876` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.701 |
| caption_frames | 25.854 |
| sample_fps | 2.036 |
| detect_object_yolo | 7.012 |
| audio_scan | 9.651 |
| asr_timings | 12.665 |
| ast_timings | 18.080 |
| describe_scenes | 6.899 |
| summarize_scenes | 11.058 |
| synthesize_synopsis | 36.192 |
| make_embedding | 2.548 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.561 |
| branch_yolo_total | 9.053 |
| branch_audio_total | 40.404 |
