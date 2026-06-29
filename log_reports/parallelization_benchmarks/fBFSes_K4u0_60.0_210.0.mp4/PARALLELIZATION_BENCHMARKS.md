# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:02:26 UTC | fBFSes_K4u0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 79.059 | 0.779 | 43.717 | 3.721 | 1.878 | 8.035 | 1.539 |

## 2026-06-26 04:02:26 UTC | fBFSes_K4u0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fBFSes_K4u0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `79.059` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 0.153 |
| caption_frames | 9.761 |
| sample_fps | 1.811 |
| detect_object_yolo | 6.282 |
| audio_scan | 10.872 |
| asr_timings | 25.809 |
| ast_timings | 7.027 |
| describe_scenes | 3.721 |
| summarize_scenes | 1.878 |
| synthesize_synopsis | 8.035 |
| make_embedding | 1.539 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 9.919 |
| branch_yolo_total | 8.098 |
| branch_audio_total | 43.717 |
