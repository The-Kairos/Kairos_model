# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:42:42 UTC | Bjl5AomF9BY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.220 | 0.790 | 61.714 | 18.264 | 16.470 | 14.200 | 5.063 |

## 2026-06-24 19:42:42 UTC | Bjl5AomF9BY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Bjl5AomF9BY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.220` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 1.877 |
| caption_frames | 52.447 |
| sample_fps | 2.541 |
| detect_object_yolo | 10.436 |
| audio_scan | 13.967 |
| asr_timings | 6.959 |
| ast_timings | 40.779 |
| describe_scenes | 18.264 |
| summarize_scenes | 16.470 |
| synthesize_synopsis | 14.200 |
| make_embedding | 5.063 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.330 |
| branch_yolo_total | 12.982 |
| branch_audio_total | 61.714 |
