# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:52:12 UTC | iy6kh6tBCmI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.254 | 0.783 | 66.769 | 21.473 | 10.997 | 26.064 | 4.114 |

## 2026-06-26 09:52:12 UTC | iy6kh6tBCmI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iy6kh6tBCmI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.254` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 1.234 |
| caption_frames | 46.867 |
| sample_fps | 2.380 |
| detect_object_yolo | 10.071 |
| audio_scan | 10.906 |
| asr_timings | 19.980 |
| ast_timings | 35.875 |
| describe_scenes | 21.473 |
| summarize_scenes | 10.997 |
| synthesize_synopsis | 26.064 |
| make_embedding | 4.114 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.108 |
| branch_yolo_total | 12.457 |
| branch_audio_total | 66.769 |
