# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:08:26 UTC | mnSYg_uCdtE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.582 | 0.784 | 60.516 | 12.941 | 21.513 | 11.797 | 4.323 |

## 2026-06-27 16:08:26 UTC | mnSYg_uCdtE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mnSYg_uCdtE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.582` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.784 |
| save_clips | - |
| sample_frames | 1.255 |
| caption_frames | 28.980 |
| sample_fps | 2.402 |
| detect_object_yolo | 9.076 |
| audio_scan | 15.338 |
| asr_timings | 10.388 |
| ast_timings | 34.783 |
| describe_scenes | 12.941 |
| summarize_scenes | 21.513 |
| synthesize_synopsis | 11.797 |
| make_embedding | 4.323 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.240 |
| branch_yolo_total | 11.484 |
| branch_audio_total | 60.516 |
