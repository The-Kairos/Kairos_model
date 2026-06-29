# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:21:48 UTC | Sl0DKauc_oU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.131 | 0.787 | 48.009 | 11.478 | 10.091 | 14.763 | 3.119 |

## 2026-06-25 17:21:48 UTC | Sl0DKauc_oU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Sl0DKauc_oU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.131` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 0.939 |
| caption_frames | 33.065 |
| sample_fps | 2.198 |
| detect_object_yolo | 8.298 |
| audio_scan | 11.937 |
| asr_timings | 12.045 |
| ast_timings | 24.018 |
| describe_scenes | 11.478 |
| summarize_scenes | 10.091 |
| synthesize_synopsis | 14.763 |
| make_embedding | 3.119 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.010 |
| branch_yolo_total | 10.502 |
| branch_audio_total | 48.009 |
