# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:26:59 UTC | 2BzoQ31IPhk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 76.093 | 1.652 | 34.193 | 4.031 | 4.175 | 5.998 | 1.603 |
| 2026-06-21 20:54:16 UTC | 2BzoQ31IPhk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-22 13:42:42 UTC | 2BzoQ31IPhk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.238 | 1.677 | 34.170 | 24.764 | 8.391 | 19.414 | 1.699 |

## 2026-06-21 09:26:59 UTC | 2BzoQ31IPhk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2BzoQ31IPhk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `76.093` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.652 |
| save_clips | - |
| sample_frames | 0.505 |
| caption_frames | 11.835 |
| sample_fps | 4.864 |
| detect_object_yolo | 5.939 |
| audio_scan | 15.936 |
| asr_timings | 11.121 |
| ast_timings | 7.127 |
| describe_scenes | 4.031 |
| summarize_scenes | 4.175 |
| synthesize_synopsis | 5.998 |
| make_embedding | 1.603 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.346 |
| branch_yolo_total | 10.808 |
| branch_audio_total | 34.193 |

## 2026-06-21 20:54:16 UTC | 2BzoQ31IPhk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2BzoQ31IPhk_360.0_510.0.mp4`
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

## 2026-06-22 13:42:42 UTC | 2BzoQ31IPhk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2BzoQ31IPhk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.238` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.677 |
| save_clips | - |
| sample_frames | 0.509 |
| caption_frames | 12.050 |
| sample_fps | 4.979 |
| detect_object_yolo | 6.164 |
| audio_scan | 16.217 |
| asr_timings | 10.724 |
| ast_timings | 7.221 |
| describe_scenes | 24.764 |
| summarize_scenes | 8.391 |
| synthesize_synopsis | 19.414 |
| make_embedding | 1.699 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.565 |
| branch_yolo_total | 11.149 |
| branch_audio_total | 34.170 |
