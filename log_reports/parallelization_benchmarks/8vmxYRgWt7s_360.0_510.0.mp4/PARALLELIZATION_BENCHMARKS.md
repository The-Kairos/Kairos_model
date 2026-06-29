# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:30:39 UTC | 8vmxYRgWt7s_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.174 | 0.698 | 65.817 | 13.958 | 13.067 | 23.580 | 4.435 |

## 2026-06-24 17:30:39 UTC | 8vmxYRgWt7s_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8vmxYRgWt7s_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.174` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.698 |
| save_clips | - |
| sample_frames | 1.466 |
| caption_frames | 49.363 |
| sample_fps | 2.349 |
| detect_object_yolo | 10.045 |
| audio_scan | 16.064 |
| asr_timings | 12.190 |
| ast_timings | 37.554 |
| describe_scenes | 13.958 |
| summarize_scenes | 13.067 |
| synthesize_synopsis | 23.580 |
| make_embedding | 4.435 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.834 |
| branch_yolo_total | 12.400 |
| branch_audio_total | 65.817 |
