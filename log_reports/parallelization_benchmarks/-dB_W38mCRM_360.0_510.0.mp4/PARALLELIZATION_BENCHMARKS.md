# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:34:37 UTC | -dB_W38mCRM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 215.288 | 1.334 | 77.927 | 13.447 | 17.112 | 10.198 | 2.741 |
| 2026-06-21 09:03:50 UTC | -dB_W38mCRM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-21 20:53:33 UTC | -dB_W38mCRM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 11:49:49 UTC | -dB_W38mCRM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 235.027 | 1.332 | 78.151 | 24.152 | 16.166 | 15.943 | 6.597 |

## 2026-06-19 22:34:37 UTC | -dB_W38mCRM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `215.288` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.334 |
| save_clips | - |
| sample_frames | 3.511 |
| caption_frames | 68.647 |
| sample_fps | 5.893 |
| detect_object_yolo | 13.091 |
| audio_scan | 11.493 |
| asr_timings | 12.938 |
| ast_timings | 53.488 |
| describe_scenes | 13.447 |
| summarize_scenes | 17.112 |
| synthesize_synopsis | 10.198 |
| make_embedding | 2.741 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 72.164 |
| branch_yolo_total | 18.990 |
| branch_audio_total | 77.927 |

## 2026-06-21 09:03:50 UTC | -dB_W38mCRM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_360.0_510.0.mp4`
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

## 2026-06-21 20:53:33 UTC | -dB_W38mCRM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_360.0_510.0.mp4`
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

## 2026-06-22 11:49:49 UTC | -dB_W38mCRM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `235.027` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.332 |
| save_clips | - |
| sample_frames | 3.541 |
| caption_frames | 68.760 |
| sample_fps | 5.975 |
| detect_object_yolo | 13.004 |
| audio_scan | 11.362 |
| asr_timings | 12.858 |
| ast_timings | 53.923 |
| describe_scenes | 24.152 |
| summarize_scenes | 16.166 |
| synthesize_synopsis | 15.943 |
| make_embedding | 6.597 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 72.306 |
| branch_yolo_total | 18.985 |
| branch_audio_total | 78.151 |
