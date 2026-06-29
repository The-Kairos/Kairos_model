# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:54:58 UTC | v-h_XnpqQ2k_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.366 | 0.689 | 54.382 | 11.286 | 9.078 | 9.165 | 3.279 |

## 2026-06-27 01:54:58 UTC | v-h_XnpqQ2k_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/v-h_XnpqQ2k_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.366` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.689 |
| save_clips | - |
| sample_frames | 1.013 |
| caption_frames | 37.001 |
| sample_fps | 2.097 |
| detect_object_yolo | 8.866 |
| audio_scan | 16.371 |
| asr_timings | 10.770 |
| ast_timings | 27.233 |
| describe_scenes | 11.286 |
| summarize_scenes | 9.078 |
| synthesize_synopsis | 9.165 |
| make_embedding | 3.279 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.020 |
| branch_yolo_total | 10.969 |
| branch_audio_total | 54.382 |
