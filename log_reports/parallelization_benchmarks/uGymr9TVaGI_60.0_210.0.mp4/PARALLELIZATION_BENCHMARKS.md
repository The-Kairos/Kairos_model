# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:52:37 UTC | uGymr9TVaGI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 95.757 | 0.771 | 37.995 | 5.835 | 9.990 | 11.639 | 2.021 |

## 2026-06-27 00:52:37 UTC | uGymr9TVaGI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uGymr9TVaGI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `95.757` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.771 |
| save_clips | - |
| sample_frames | 0.393 |
| caption_frames | 17.443 |
| sample_fps | 1.902 |
| detect_object_yolo | 6.336 |
| audio_scan | 15.830 |
| asr_timings | 8.391 |
| ast_timings | 13.764 |
| describe_scenes | 5.835 |
| summarize_scenes | 9.990 |
| synthesize_synopsis | 11.639 |
| make_embedding | 2.021 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.842 |
| branch_yolo_total | 8.243 |
| branch_audio_total | 37.995 |
