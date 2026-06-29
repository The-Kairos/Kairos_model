# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:50:47 UTC | v-h_XnpqQ2k_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.347 | 0.684 | 53.938 | 10.635 | 8.798 | 20.105 | 3.858 |

## 2026-06-27 01:50:47 UTC | v-h_XnpqQ2k_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/v-h_XnpqQ2k_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.347` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 1.127 |
| caption_frames | 43.172 |
| sample_fps | 2.139 |
| detect_object_yolo | 9.450 |
| audio_scan | 11.831 |
| asr_timings | 9.230 |
| ast_timings | 32.868 |
| describe_scenes | 10.635 |
| summarize_scenes | 8.798 |
| synthesize_synopsis | 20.105 |
| make_embedding | 3.858 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.306 |
| branch_yolo_total | 11.594 |
| branch_audio_total | 53.938 |
