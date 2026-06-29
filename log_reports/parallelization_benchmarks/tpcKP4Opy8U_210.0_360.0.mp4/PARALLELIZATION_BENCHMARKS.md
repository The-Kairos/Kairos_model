# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:54:52 UTC | tpcKP4Opy8U_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.995 | 0.631 | 54.312 | 12.053 | 8.437 | 9.056 | 3.522 |

## 2026-06-26 23:54:52 UTC | tpcKP4Opy8U_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tpcKP4Opy8U_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.995` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.631 |
| save_clips | - |
| sample_frames | 1.257 |
| caption_frames | 39.310 |
| sample_fps | 2.132 |
| detect_object_yolo | 8.846 |
| audio_scan | 12.834 |
| asr_timings | 10.890 |
| ast_timings | 30.580 |
| describe_scenes | 12.053 |
| summarize_scenes | 8.437 |
| synthesize_synopsis | 9.056 |
| make_embedding | 3.522 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.572 |
| branch_yolo_total | 10.984 |
| branch_audio_total | 54.312 |
