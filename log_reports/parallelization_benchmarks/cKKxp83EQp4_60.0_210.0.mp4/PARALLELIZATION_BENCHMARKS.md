# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:27:10 UTC | cKKxp83EQp4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 68.109 | 0.810 | 15.773 | 7.074 | 5.034 | 13.921 | 1.791 |

## 2026-06-26 02:27:10 UTC | cKKxp83EQp4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/cKKxp83EQp4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `68.109` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 0.394 |
| caption_frames | 15.373 |
| sample_fps | 1.896 |
| detect_object_yolo | 6.329 |
| audio_scan | 3.857 |
| asr_timings | 0.000 |
| ast_timings | 10.220 |
| describe_scenes | 7.074 |
| summarize_scenes | 5.034 |
| synthesize_synopsis | 13.921 |
| make_embedding | 1.791 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.773 |
| branch_yolo_total | 8.231 |
| branch_audio_total | 14.086 |
