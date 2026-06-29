# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:20:06 UTC | 23YBs2JxE-k_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 130.948 | 1.962 | 50.643 | 6.685 | 8.529 | 8.348 | 3.404 |
| 2026-06-21 20:54:12 UTC | 23YBs2JxE-k_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 13:32:41 UTC | 23YBs2JxE-k_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.055 | 1.998 | 48.303 | 22.044 | 15.977 | 21.993 | 3.301 |

## 2026-06-21 09:20:06 UTC | 23YBs2JxE-k_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/23YBs2JxE-k_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `130.948` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.962 |
| save_clips | - |
| sample_frames | 2.582 |
| caption_frames | 33.814 |
| sample_fps | 5.849 |
| detect_object_yolo | 7.842 |
| audio_scan | 12.651 |
| asr_timings | 11.825 |
| ast_timings | 26.159 |
| describe_scenes | 6.685 |
| summarize_scenes | 8.529 |
| synthesize_synopsis | 8.348 |
| make_embedding | 3.404 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.402 |
| branch_yolo_total | 13.697 |
| branch_audio_total | 50.643 |

## 2026-06-21 20:54:12 UTC | 23YBs2JxE-k_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/23YBs2JxE-k_210.0_360.0.mp4`
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

## 2026-06-22 13:32:41 UTC | 23YBs2JxE-k_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/23YBs2JxE-k_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.055` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.998 |
| save_clips | - |
| sample_frames | 2.628 |
| caption_frames | 33.404 |
| sample_fps | 5.925 |
| detect_object_yolo | 8.109 |
| audio_scan | 12.892 |
| asr_timings | 8.899 |
| ast_timings | 26.504 |
| describe_scenes | 22.044 |
| summarize_scenes | 15.977 |
| synthesize_synopsis | 21.993 |
| make_embedding | 3.301 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.038 |
| branch_yolo_total | 14.040 |
| branch_audio_total | 48.303 |
