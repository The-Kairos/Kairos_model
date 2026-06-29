# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:53:03 UTC | pKGH1tyyCcY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 216.673 | 0.787 | 102.099 | 13.303 | 16.306 | 9.093 | 5.047 |

## 2026-06-28 07:53:03 UTC | pKGH1tyyCcY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pKGH1tyyCcY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `216.673` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.803 |
| caption_frames | 52.980 |
| sample_fps | 2.638 |
| detect_object_yolo | 11.214 |
| audio_scan | 8.498 |
| asr_timings | 52.305 |
| ast_timings | 41.287 |
| describe_scenes | 13.303 |
| summarize_scenes | 16.306 |
| synthesize_synopsis | 9.093 |
| make_embedding | 5.047 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.789 |
| branch_yolo_total | 13.858 |
| branch_audio_total | 102.099 |
