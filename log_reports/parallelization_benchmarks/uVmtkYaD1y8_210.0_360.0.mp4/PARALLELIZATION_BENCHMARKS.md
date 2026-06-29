# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:56:21 UTC | uVmtkYaD1y8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.472 | 0.634 | 62.899 | 10.531 | 9.571 | 7.676 | 4.063 |

## 2026-06-27 00:56:21 UTC | uVmtkYaD1y8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uVmtkYaD1y8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.472` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.634 |
| save_clips | - |
| sample_frames | 1.205 |
| caption_frames | 46.850 |
| sample_fps | 2.129 |
| detect_object_yolo | 9.493 |
| audio_scan | 15.991 |
| asr_timings | 10.886 |
| ast_timings | 36.013 |
| describe_scenes | 10.531 |
| summarize_scenes | 9.571 |
| synthesize_synopsis | 7.676 |
| make_embedding | 4.063 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.061 |
| branch_yolo_total | 11.627 |
| branch_audio_total | 62.899 |
