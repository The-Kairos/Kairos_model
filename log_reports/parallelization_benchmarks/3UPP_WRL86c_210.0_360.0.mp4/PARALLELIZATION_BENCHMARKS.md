# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:07:28 UTC | 3UPP_WRL86c_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.591 | 0.773 | 78.235 | 9.969 | 5.688 | 6.611 | 3.873 |

## 2026-06-21 22:07:28 UTC | 3UPP_WRL86c_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3UPP_WRL86c_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.591` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 1.176 |
| caption_frames | 43.219 |
| sample_fps | 2.322 |
| detect_object_yolo | 9.307 |
| audio_scan | 12.878 |
| asr_timings | 32.564 |
| ast_timings | 32.784 |
| describe_scenes | 9.969 |
| summarize_scenes | 5.688 |
| synthesize_synopsis | 6.611 |
| make_embedding | 3.873 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.401 |
| branch_yolo_total | 11.635 |
| branch_audio_total | 78.235 |
