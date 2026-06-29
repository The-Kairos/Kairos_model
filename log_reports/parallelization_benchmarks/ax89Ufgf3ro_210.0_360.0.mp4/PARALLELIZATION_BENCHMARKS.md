# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:27:53 UTC | ax89Ufgf3ro_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.732 | 0.650 | 79.721 | 22.441 | 8.466 | 14.924 | 4.490 |

## 2026-06-26 00:27:53 UTC | ax89Ufgf3ro_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ax89Ufgf3ro_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.732` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.650 |
| save_clips | - |
| sample_frames | 1.917 |
| caption_frames | 48.943 |
| sample_fps | 2.402 |
| detect_object_yolo | 10.347 |
| audio_scan | 16.063 |
| asr_timings | 26.212 |
| ast_timings | 37.437 |
| describe_scenes | 22.441 |
| summarize_scenes | 8.466 |
| synthesize_synopsis | 14.924 |
| make_embedding | 4.490 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.866 |
| branch_yolo_total | 12.755 |
| branch_audio_total | 79.721 |
