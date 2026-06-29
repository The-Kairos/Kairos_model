# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:53:34 UTC | xOJWLSOp3zU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.439 | 0.780 | 54.149 | 12.844 | 5.696 | 7.814 | 3.654 |

## 2026-06-27 03:53:34 UTC | xOJWLSOp3zU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xOJWLSOp3zU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.439` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.178 |
| caption_frames | 39.711 |
| sample_fps | 2.285 |
| detect_object_yolo | 8.850 |
| audio_scan | 14.275 |
| asr_timings | 9.821 |
| ast_timings | 30.045 |
| describe_scenes | 12.844 |
| summarize_scenes | 5.696 |
| synthesize_synopsis | 7.814 |
| make_embedding | 3.654 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.895 |
| branch_yolo_total | 11.141 |
| branch_audio_total | 54.149 |
