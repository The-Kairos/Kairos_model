# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:46:12 UTC | Rzu3oZyGzKw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 201.392 | 0.654 | 85.075 | 14.808 | 19.561 | 14.080 | 4.237 |

## 2026-06-25 16:46:12 UTC | Rzu3oZyGzKw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Rzu3oZyGzKw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `201.392` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.654 |
| save_clips | - |
| sample_frames | 1.188 |
| caption_frames | 48.196 |
| sample_fps | 2.268 |
| detect_object_yolo | 9.911 |
| audio_scan | 13.404 |
| asr_timings | 35.588 |
| ast_timings | 36.075 |
| describe_scenes | 14.808 |
| summarize_scenes | 19.561 |
| synthesize_synopsis | 14.080 |
| make_embedding | 4.237 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.389 |
| branch_yolo_total | 12.185 |
| branch_audio_total | 85.075 |
