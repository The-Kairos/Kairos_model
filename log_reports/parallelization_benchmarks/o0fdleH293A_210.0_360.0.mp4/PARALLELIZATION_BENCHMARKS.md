# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:58:18 UTC | o0fdleH293A_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.300 | 0.767 | 62.135 | 11.401 | 8.321 | 7.713 | 4.203 |

## 2026-06-27 16:58:18 UTC | o0fdleH293A_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/o0fdleH293A_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.300` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.176 |
| caption_frames | 47.353 |
| sample_fps | 2.342 |
| detect_object_yolo | 9.506 |
| audio_scan | 14.813 |
| asr_timings | 11.889 |
| ast_timings | 35.425 |
| describe_scenes | 11.401 |
| summarize_scenes | 8.321 |
| synthesize_synopsis | 7.713 |
| make_embedding | 4.203 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.535 |
| branch_yolo_total | 11.855 |
| branch_audio_total | 62.135 |
