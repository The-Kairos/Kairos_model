# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:11:16 UTC | zpRzgcjq0x8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.799 | 0.707 | 60.283 | 13.955 | 8.377 | 8.670 | 5.030 |

## 2026-06-27 06:11:16 UTC | zpRzgcjq0x8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zpRzgcjq0x8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.799` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.707 |
| save_clips | - |
| sample_frames | 1.461 |
| caption_frames | 54.606 |
| sample_fps | 2.387 |
| detect_object_yolo | 10.902 |
| audio_scan | 11.826 |
| asr_timings | 6.914 |
| ast_timings | 41.535 |
| describe_scenes | 13.955 |
| summarize_scenes | 8.377 |
| synthesize_synopsis | 8.670 |
| make_embedding | 5.030 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.074 |
| branch_yolo_total | 13.295 |
| branch_audio_total | 60.283 |
