# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 08:14:21 UTC | MH2nvDD_uo0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.967 | 0.777 | 91.559 | 17.577 | 23.937 | 22.898 | 3.057 |

## 2026-06-25 08:14:21 UTC | MH2nvDD_uo0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MH2nvDD_uo0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.967` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 0.914 |
| caption_frames | 32.174 |
| sample_fps | 2.153 |
| detect_object_yolo | 8.506 |
| audio_scan | 15.899 |
| asr_timings | 52.344 |
| ast_timings | 23.307 |
| describe_scenes | 17.577 |
| summarize_scenes | 23.937 |
| synthesize_synopsis | 22.898 |
| make_embedding | 3.057 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.093 |
| branch_yolo_total | 10.665 |
| branch_audio_total | 91.559 |
