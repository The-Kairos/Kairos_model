# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:23:34 UTC | 23YBs2JxE-k_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.868 | 1.933 | 48.154 | 6.820 | 9.565 | 7.514 | 3.052 |
| 2026-06-21 20:54:14 UTC | 23YBs2JxE-k_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 13:37:36 UTC | 23YBs2JxE-k_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.458 | 1.947 | 50.008 | 17.094 | 14.272 | 32.916 | 3.014 |

## 2026-06-21 09:23:34 UTC | 23YBs2JxE-k_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/23YBs2JxE-k_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.868` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.933 |
| save_clips | - |
| sample_frames | 2.346 |
| caption_frames | 29.994 |
| sample_fps | 5.838 |
| detect_object_yolo | 7.367 |
| audio_scan | 14.772 |
| asr_timings | 10.197 |
| ast_timings | 23.177 |
| describe_scenes | 6.820 |
| summarize_scenes | 9.565 |
| synthesize_synopsis | 7.514 |
| make_embedding | 3.052 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.346 |
| branch_yolo_total | 13.211 |
| branch_audio_total | 48.154 |

## 2026-06-21 20:54:14 UTC | 23YBs2JxE-k_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/23YBs2JxE-k_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.060` sec

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

## 2026-06-22 13:37:36 UTC | 23YBs2JxE-k_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/23YBs2JxE-k_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.458` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.947 |
| save_clips | - |
| sample_frames | 2.363 |
| caption_frames | 30.939 |
| sample_fps | 5.867 |
| detect_object_yolo | 7.636 |
| audio_scan | 14.897 |
| asr_timings | 11.174 |
| ast_timings | 23.929 |
| describe_scenes | 17.094 |
| summarize_scenes | 14.272 |
| synthesize_synopsis | 32.916 |
| make_embedding | 3.014 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.308 |
| branch_yolo_total | 13.509 |
| branch_audio_total | 50.008 |
