# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:28:45 UTC | olnRUozUO5M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.245 | 0.818 | 73.870 | 17.836 | 8.842 | 7.747 | 6.094 |

## 2026-06-28 07:28:45 UTC | olnRUozUO5M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/olnRUozUO5M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.245` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 1.765 |
| caption_frames | 64.269 |
| sample_fps | 2.682 |
| detect_object_yolo | 11.899 |
| audio_scan | 14.892 |
| asr_timings | 9.629 |
| ast_timings | 49.341 |
| describe_scenes | 17.836 |
| summarize_scenes | 8.842 |
| synthesize_synopsis | 7.747 |
| make_embedding | 6.094 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 66.040 |
| branch_yolo_total | 14.587 |
| branch_audio_total | 73.870 |
