# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:48:10 UTC | utW1ItcMeJw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.574 | 0.816 | 72.840 | 11.986 | 9.690 | 8.840 | 5.748 |

## 2026-06-27 01:48:10 UTC | utW1ItcMeJw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/utW1ItcMeJw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.574` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 1.344 |
| caption_frames | 63.240 |
| sample_fps | 2.559 |
| detect_object_yolo | 13.013 |
| audio_scan | 15.308 |
| asr_timings | 11.910 |
| ast_timings | 45.613 |
| describe_scenes | 11.986 |
| summarize_scenes | 9.690 |
| synthesize_synopsis | 8.840 |
| make_embedding | 5.748 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 64.590 |
| branch_yolo_total | 15.578 |
| branch_audio_total | 72.840 |
