# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:43:53 UTC | hj9Vh2NagqA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.144 | 0.828 | 64.781 | 15.935 | 25.475 | 20.829 | 4.045 |

## 2026-06-26 06:43:53 UTC | hj9Vh2NagqA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hj9Vh2NagqA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.144` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.828 |
| save_clips | - |
| sample_frames | 1.353 |
| caption_frames | 42.056 |
| sample_fps | 2.410 |
| detect_object_yolo | 9.020 |
| audio_scan | 15.091 |
| asr_timings | 17.278 |
| ast_timings | 32.404 |
| describe_scenes | 15.935 |
| summarize_scenes | 25.475 |
| synthesize_synopsis | 20.829 |
| make_embedding | 4.045 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.415 |
| branch_yolo_total | 11.436 |
| branch_audio_total | 64.781 |
