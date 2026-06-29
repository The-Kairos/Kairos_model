# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:43:22 UTC | LmnfadjuaE8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.683 | 0.628 | 53.013 | 16.983 | 18.517 | 19.457 | 3.890 |

## 2026-06-25 07:43:22 UTC | LmnfadjuaE8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LmnfadjuaE8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.683` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.628 |
| save_clips | - |
| sample_frames | 1.231 |
| caption_frames | 43.843 |
| sample_fps | 2.195 |
| detect_object_yolo | 9.476 |
| audio_scan | 16.047 |
| asr_timings | 7.795 |
| ast_timings | 29.163 |
| describe_scenes | 16.983 |
| summarize_scenes | 18.517 |
| synthesize_synopsis | 19.457 |
| make_embedding | 3.890 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.080 |
| branch_yolo_total | 11.677 |
| branch_audio_total | 53.013 |
