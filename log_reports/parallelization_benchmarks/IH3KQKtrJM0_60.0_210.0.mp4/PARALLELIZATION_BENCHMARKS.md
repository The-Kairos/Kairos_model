# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:45:12 UTC | IH3KQKtrJM0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 122.768 | 0.830 | 44.653 | 11.888 | 6.672 | 16.440 | 2.763 |

## 2026-06-25 04:45:12 UTC | IH3KQKtrJM0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IH3KQKtrJM0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `122.768` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.830 |
| save_clips | - |
| sample_frames | 0.791 |
| caption_frames | 28.008 |
| sample_fps | 2.089 |
| detect_object_yolo | 7.230 |
| audio_scan | 14.860 |
| asr_timings | 9.770 |
| ast_timings | 20.015 |
| describe_scenes | 11.888 |
| summarize_scenes | 6.672 |
| synthesize_synopsis | 16.440 |
| make_embedding | 2.763 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.805 |
| branch_yolo_total | 9.325 |
| branch_audio_total | 44.653 |
