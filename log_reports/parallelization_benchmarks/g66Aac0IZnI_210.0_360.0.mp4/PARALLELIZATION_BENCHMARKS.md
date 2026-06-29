# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:49:03 UTC | g66Aac0IZnI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.933 | 0.859 | 57.195 | 14.868 | 16.262 | 18.668 | 4.169 |

## 2026-06-26 04:49:03 UTC | g66Aac0IZnI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/g66Aac0IZnI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.933` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.859 |
| save_clips | - |
| sample_frames | 1.446 |
| caption_frames | 45.776 |
| sample_fps | 2.507 |
| detect_object_yolo | 9.749 |
| audio_scan | 14.041 |
| asr_timings | 7.172 |
| ast_timings | 35.974 |
| describe_scenes | 14.868 |
| summarize_scenes | 16.262 |
| synthesize_synopsis | 18.668 |
| make_embedding | 4.169 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.228 |
| branch_yolo_total | 12.262 |
| branch_audio_total | 57.195 |
