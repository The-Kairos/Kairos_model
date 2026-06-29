# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:00:27 UTC | G19tOR4S2fM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.999 | 0.693 | 46.268 | 9.764 | 11.004 | 9.972 | 3.039 |

## 2026-06-25 01:00:27 UTC | G19tOR4S2fM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G19tOR4S2fM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.999` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.693 |
| save_clips | - |
| sample_frames | 0.767 |
| caption_frames | 31.840 |
| sample_fps | 1.969 |
| detect_object_yolo | 8.225 |
| audio_scan | 13.845 |
| asr_timings | 8.291 |
| ast_timings | 24.124 |
| describe_scenes | 9.764 |
| summarize_scenes | 11.004 |
| synthesize_synopsis | 9.972 |
| make_embedding | 3.039 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.613 |
| branch_yolo_total | 10.201 |
| branch_audio_total | 46.268 |
