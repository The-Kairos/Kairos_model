# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:36:26 UTC | 9J4LmsquLec_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 81.541 | 0.779 | 33.094 | 4.598 | 4.741 | 16.130 | 1.600 |

## 2026-06-24 17:36:26 UTC | 9J4LmsquLec_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9J4LmsquLec_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `81.541` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 0.161 |
| caption_frames | 11.072 |
| sample_fps | 1.773 |
| detect_object_yolo | 6.208 |
| audio_scan | 16.025 |
| asr_timings | 9.808 |
| ast_timings | 7.252 |
| describe_scenes | 4.598 |
| summarize_scenes | 4.741 |
| synthesize_synopsis | 16.130 |
| make_embedding | 1.600 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 11.239 |
| branch_yolo_total | 7.987 |
| branch_audio_total | 33.094 |
