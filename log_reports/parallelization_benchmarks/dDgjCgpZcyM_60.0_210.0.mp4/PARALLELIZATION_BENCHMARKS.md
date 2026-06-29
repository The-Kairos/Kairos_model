# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:52:04 UTC | dDgjCgpZcyM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 104.819 | 0.637 | 27.527 | 5.471 | 26.627 | 21.866 | 1.598 |

## 2026-06-26 02:52:04 UTC | dDgjCgpZcyM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dDgjCgpZcyM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `104.819` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.637 |
| save_clips | - |
| sample_frames | 0.195 |
| caption_frames | 11.897 |
| sample_fps | 1.631 |
| detect_object_yolo | 5.971 |
| audio_scan | 8.706 |
| asr_timings | 11.508 |
| ast_timings | 7.303 |
| describe_scenes | 5.471 |
| summarize_scenes | 26.627 |
| synthesize_synopsis | 21.866 |
| make_embedding | 1.598 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.099 |
| branch_yolo_total | 7.608 |
| branch_audio_total | 27.527 |
