# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 20:06:46 UTC | sjpkuDXeSBM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.831 | 0.800 | 81.192 | 17.395 | 10.091 | 25.478 | 3.857 |

## 2026-06-26 20:06:46 UTC | sjpkuDXeSBM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sjpkuDXeSBM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.831` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 1.123 |
| caption_frames | 42.094 |
| sample_fps | 2.305 |
| detect_object_yolo | 9.067 |
| audio_scan | 15.089 |
| asr_timings | 33.361 |
| ast_timings | 32.734 |
| describe_scenes | 17.395 |
| summarize_scenes | 10.091 |
| synthesize_synopsis | 25.478 |
| make_embedding | 3.857 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.223 |
| branch_yolo_total | 11.378 |
| branch_audio_total | 81.192 |
