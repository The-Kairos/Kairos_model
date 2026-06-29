# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:58:46 UTC | AS6BctRAyU4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 109.203 | 0.773 | 41.683 | 8.307 | 6.419 | 14.673 | 2.508 |

## 2026-06-24 18:58:46 UTC | AS6BctRAyU4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AS6BctRAyU4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `109.203` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 0.485 |
| caption_frames | 23.076 |
| sample_fps | 1.986 |
| detect_object_yolo | 7.890 |
| audio_scan | 13.886 |
| asr_timings | 9.288 |
| ast_timings | 18.501 |
| describe_scenes | 8.307 |
| summarize_scenes | 6.419 |
| synthesize_synopsis | 14.673 |
| make_embedding | 2.508 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.567 |
| branch_yolo_total | 9.882 |
| branch_audio_total | 41.683 |
