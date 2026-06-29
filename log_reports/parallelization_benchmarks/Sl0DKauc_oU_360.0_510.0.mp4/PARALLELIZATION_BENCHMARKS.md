# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:24:59 UTC | Sl0DKauc_oU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.981 | 0.792 | 55.437 | 14.503 | 31.750 | 20.516 | 4.158 |

## 2026-06-25 17:24:59 UTC | Sl0DKauc_oU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Sl0DKauc_oU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.981` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.264 |
| caption_frames | 47.613 |
| sample_fps | 2.315 |
| detect_object_yolo | 10.131 |
| audio_scan | 8.665 |
| asr_timings | 11.655 |
| ast_timings | 35.109 |
| describe_scenes | 14.503 |
| summarize_scenes | 31.750 |
| synthesize_synopsis | 20.516 |
| make_embedding | 4.158 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.883 |
| branch_yolo_total | 12.451 |
| branch_audio_total | 55.437 |
