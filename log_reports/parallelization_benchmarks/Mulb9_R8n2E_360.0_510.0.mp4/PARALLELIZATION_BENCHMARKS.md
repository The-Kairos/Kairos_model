# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:03:44 UTC | Mulb9_R8n2E_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 117.218 | 1.613 | 61.459 | 5.599 | 8.762 | 15.784 | 1.598 |

## 2026-06-25 10:03:44 UTC | Mulb9_R8n2E_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Mulb9_R8n2E_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `117.218` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.613 |
| save_clips | - |
| sample_frames | 0.260 |
| caption_frames | 12.422 |
| sample_fps | 1.840 |
| detect_object_yolo | 6.442 |
| audio_scan | 15.943 |
| asr_timings | 38.221 |
| ast_timings | 7.286 |
| describe_scenes | 5.599 |
| summarize_scenes | 8.762 |
| synthesize_synopsis | 15.784 |
| make_embedding | 1.598 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.687 |
| branch_yolo_total | 8.287 |
| branch_audio_total | 61.459 |
