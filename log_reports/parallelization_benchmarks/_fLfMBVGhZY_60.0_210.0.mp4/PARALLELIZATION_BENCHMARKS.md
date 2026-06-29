# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:04:12 UTC | _fLfMBVGhZY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.834 | 0.790 | 50.929 | 8.866 | 15.002 | 15.353 | 3.287 |

## 2026-06-26 00:04:12 UTC | _fLfMBVGhZY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_fLfMBVGhZY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.834` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 0.970 |
| caption_frames | 33.927 |
| sample_fps | 2.154 |
| detect_object_yolo | 8.154 |
| audio_scan | 13.806 |
| asr_timings | 10.460 |
| ast_timings | 26.656 |
| describe_scenes | 8.866 |
| summarize_scenes | 15.002 |
| synthesize_synopsis | 15.353 |
| make_embedding | 3.287 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.903 |
| branch_yolo_total | 10.314 |
| branch_audio_total | 50.929 |
