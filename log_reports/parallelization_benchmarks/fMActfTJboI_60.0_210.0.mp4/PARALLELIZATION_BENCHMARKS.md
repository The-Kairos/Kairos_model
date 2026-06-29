# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:14:19 UTC | fMActfTJboI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.213 | 0.793 | 50.505 | 8.691 | 9.946 | 11.621 | 3.026 |

## 2026-06-26 04:14:19 UTC | fMActfTJboI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fMActfTJboI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.213` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 0.839 |
| caption_frames | 30.992 |
| sample_fps | 2.177 |
| detect_object_yolo | 8.169 |
| audio_scan | 14.167 |
| asr_timings | 11.886 |
| ast_timings | 24.443 |
| describe_scenes | 8.691 |
| summarize_scenes | 9.946 |
| synthesize_synopsis | 11.621 |
| make_embedding | 3.026 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.837 |
| branch_yolo_total | 10.352 |
| branch_audio_total | 50.505 |
