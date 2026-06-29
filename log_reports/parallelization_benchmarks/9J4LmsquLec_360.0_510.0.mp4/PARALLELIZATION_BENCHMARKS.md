# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:38:16 UTC | 9J4LmsquLec_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 109.384 | 0.794 | 31.249 | 4.395 | 5.713 | 44.120 | 1.565 |

## 2026-06-24 17:38:16 UTC | 9J4LmsquLec_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9J4LmsquLec_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `109.384` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.794 |
| save_clips | - |
| sample_frames | 0.203 |
| caption_frames | 12.291 |
| sample_fps | 1.800 |
| detect_object_yolo | 5.864 |
| audio_scan | 13.910 |
| asr_timings | 9.998 |
| ast_timings | 7.332 |
| describe_scenes | 4.395 |
| summarize_scenes | 5.713 |
| synthesize_synopsis | 44.120 |
| make_embedding | 1.565 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.500 |
| branch_yolo_total | 7.670 |
| branch_audio_total | 31.249 |
