# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:27:31 UTC | 8vmxYRgWt7s_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.847 | 0.710 | 57.226 | 15.353 | 32.049 | 20.166 | 4.175 |

## 2026-06-24 17:27:31 UTC | 8vmxYRgWt7s_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8vmxYRgWt7s_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.847` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.710 |
| save_clips | - |
| sample_frames | 1.465 |
| caption_frames | 48.402 |
| sample_fps | 2.300 |
| detect_object_yolo | 9.610 |
| audio_scan | 12.723 |
| asr_timings | 9.269 |
| ast_timings | 35.226 |
| describe_scenes | 15.353 |
| summarize_scenes | 32.049 |
| synthesize_synopsis | 20.166 |
| make_embedding | 4.175 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.872 |
| branch_yolo_total | 11.916 |
| branch_audio_total | 57.226 |
