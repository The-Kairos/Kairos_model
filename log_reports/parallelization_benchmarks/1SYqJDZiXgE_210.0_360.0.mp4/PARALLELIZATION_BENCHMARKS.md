# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:27:59 UTC | 1SYqJDZiXgE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 249.239 | 0.674 | 73.060 | 30.837 | 47.735 | 19.723 | 5.080 |
| 2026-06-27 15:02:50 UTC | 1SYqJDZiXgE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.085 | 0.642 | 74.463 | 13.864 | 10.861 | 11.538 | 5.068 |

## 2026-06-23 13:27:59 UTC | 1SYqJDZiXgE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1SYqJDZiXgE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `249.239` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 1.730 |
| caption_frames | 55.595 |
| sample_fps | 2.417 |
| detect_object_yolo | 10.946 |
| audio_scan | 13.743 |
| asr_timings | 18.083 |
| ast_timings | 41.226 |
| describe_scenes | 30.837 |
| summarize_scenes | 47.735 |
| synthesize_synopsis | 19.723 |
| make_embedding | 5.080 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.331 |
| branch_yolo_total | 13.369 |
| branch_audio_total | 73.060 |

## 2026-06-27 15:02:50 UTC | 1SYqJDZiXgE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1SYqJDZiXgE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.085` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.642 |
| save_clips | - |
| sample_frames | 1.740 |
| caption_frames | 54.339 |
| sample_fps | 2.413 |
| detect_object_yolo | 10.740 |
| audio_scan | 13.872 |
| asr_timings | 18.579 |
| ast_timings | 42.004 |
| describe_scenes | 13.864 |
| summarize_scenes | 10.861 |
| synthesize_synopsis | 11.538 |
| make_embedding | 5.068 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.085 |
| branch_yolo_total | 13.158 |
| branch_audio_total | 74.463 |
