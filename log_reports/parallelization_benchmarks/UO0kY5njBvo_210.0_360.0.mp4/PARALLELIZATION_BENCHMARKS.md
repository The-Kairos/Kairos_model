# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:53:39 UTC | UO0kY5njBvo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.313 | 0.777 | 48.468 | 11.254 | 10.757 | 12.258 | 2.520 |

## 2026-06-25 18:53:39 UTC | UO0kY5njBvo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UO0kY5njBvo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.313` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 0.677 |
| caption_frames | 26.144 |
| sample_fps | 2.008 |
| detect_object_yolo | 7.042 |
| audio_scan | 13.779 |
| asr_timings | 15.672 |
| ast_timings | 19.009 |
| describe_scenes | 11.254 |
| summarize_scenes | 10.757 |
| synthesize_synopsis | 12.258 |
| make_embedding | 2.520 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.827 |
| branch_yolo_total | 9.056 |
| branch_audio_total | 48.468 |
