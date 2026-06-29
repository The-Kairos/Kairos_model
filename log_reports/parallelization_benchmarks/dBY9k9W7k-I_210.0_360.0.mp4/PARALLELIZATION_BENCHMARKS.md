# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:40:14 UTC | dBY9k9W7k-I_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 148.484 | 0.799 | 54.406 | 16.017 | 8.776 | 10.018 | 3.637 |

## 2026-06-26 02:40:14 UTC | dBY9k9W7k-I_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dBY9k9W7k-I_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `148.484` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.173 |
| caption_frames | 41.300 |
| sample_fps | 2.297 |
| detect_object_yolo | 8.663 |
| audio_scan | 7.623 |
| asr_timings | 15.980 |
| ast_timings | 30.795 |
| describe_scenes | 16.017 |
| summarize_scenes | 8.776 |
| synthesize_synopsis | 10.018 |
| make_embedding | 3.637 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.478 |
| branch_yolo_total | 10.966 |
| branch_audio_total | 54.406 |
