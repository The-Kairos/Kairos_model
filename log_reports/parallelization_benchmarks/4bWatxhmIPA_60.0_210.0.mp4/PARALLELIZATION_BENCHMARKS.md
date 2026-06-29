# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:17:41 UTC | 4bWatxhmIPA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.095 | 0.655 | 56.470 | 23.701 | 30.950 | 20.505 | 3.835 |
| 2026-06-24 11:10:24 UTC | 4bWatxhmIPA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.714 | 0.653 | 56.832 | 11.944 | 12.448 | 15.959 | 3.842 |

## 2026-06-23 17:17:41 UTC | 4bWatxhmIPA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4bWatxhmIPA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.095` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.655 |
| save_clips | - |
| sample_frames | 1.010 |
| caption_frames | 43.057 |
| sample_fps | 2.099 |
| detect_object_yolo | 9.386 |
| audio_scan | 14.818 |
| asr_timings | 10.332 |
| ast_timings | 31.311 |
| describe_scenes | 23.701 |
| summarize_scenes | 30.950 |
| synthesize_synopsis | 20.505 |
| make_embedding | 3.835 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.073 |
| branch_yolo_total | 11.491 |
| branch_audio_total | 56.470 |

## 2026-06-24 11:10:24 UTC | 4bWatxhmIPA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4bWatxhmIPA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.714` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.653 |
| save_clips | - |
| sample_frames | 1.002 |
| caption_frames | 42.346 |
| sample_fps | 2.114 |
| detect_object_yolo | 9.171 |
| audio_scan | 14.807 |
| asr_timings | 10.641 |
| ast_timings | 31.376 |
| describe_scenes | 11.944 |
| summarize_scenes | 12.448 |
| synthesize_synopsis | 15.959 |
| make_embedding | 3.842 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.353 |
| branch_yolo_total | 11.291 |
| branch_audio_total | 56.832 |
