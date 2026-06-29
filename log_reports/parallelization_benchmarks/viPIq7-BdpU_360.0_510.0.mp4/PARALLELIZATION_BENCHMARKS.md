# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:29:01 UTC | viPIq7-BdpU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.796 | 0.652 | 53.080 | 8.360 | 10.551 | 10.309 | 3.295 |

## 2026-06-27 02:29:01 UTC | viPIq7-BdpU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/viPIq7-BdpU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.796` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.652 |
| save_clips | - |
| sample_frames | 0.864 |
| caption_frames | 35.945 |
| sample_fps | 2.067 |
| detect_object_yolo | 8.265 |
| audio_scan | 16.196 |
| asr_timings | 10.117 |
| ast_timings | 26.759 |
| describe_scenes | 8.360 |
| summarize_scenes | 10.551 |
| synthesize_synopsis | 10.309 |
| make_embedding | 3.295 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.816 |
| branch_yolo_total | 10.338 |
| branch_audio_total | 53.080 |
