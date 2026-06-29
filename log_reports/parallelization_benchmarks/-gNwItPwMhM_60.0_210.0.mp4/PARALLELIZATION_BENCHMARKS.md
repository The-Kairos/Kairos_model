# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:36 UTC | -gNwItPwMhM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-22 11:58:26 UTC | -gNwItPwMhM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.417 | 0.796 | 49.655 | 17.961 | 17.222 | 18.426 | 3.603 |

## 2026-06-21 20:53:36 UTC | -gNwItPwMhM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-gNwItPwMhM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.059` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 11:58:26 UTC | -gNwItPwMhM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-gNwItPwMhM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.417` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.132 |
| caption_frames | 38.922 |
| sample_fps | 2.254 |
| detect_object_yolo | 9.058 |
| audio_scan | 10.637 |
| asr_timings | 9.436 |
| ast_timings | 29.574 |
| describe_scenes | 17.961 |
| summarize_scenes | 17.222 |
| synthesize_synopsis | 18.426 |
| make_embedding | 3.603 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.060 |
| branch_yolo_total | 11.317 |
| branch_audio_total | 49.655 |
