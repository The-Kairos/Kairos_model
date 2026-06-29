# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:53:13 UTC | 2iW3ei-5fpE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.769 | 1.462 | 45.055 | 6.020 | 5.646 | 4.837 | 2.830 |
| 2026-06-21 21:32:34 UTC | 2iW3ei-5fpE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 121.711 | 1.425 | 45.248 | 6.414 | 6.425 | 12.561 | 2.815 |

## 2026-06-21 09:53:13 UTC | 2iW3ei-5fpE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2iW3ei-5fpE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.769` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.462 |
| save_clips | - |
| sample_frames | 1.736 |
| caption_frames | 31.007 |
| sample_fps | 5.109 |
| detect_object_yolo | 7.707 |
| audio_scan | 13.861 |
| asr_timings | 9.679 |
| ast_timings | 21.506 |
| describe_scenes | 6.020 |
| summarize_scenes | 5.646 |
| synthesize_synopsis | 4.837 |
| make_embedding | 2.830 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.748 |
| branch_yolo_total | 12.823 |
| branch_audio_total | 45.055 |

## 2026-06-21 21:32:34 UTC | 2iW3ei-5fpE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2iW3ei-5fpE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `121.711` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.425 |
| save_clips | - |
| sample_frames | 1.720 |
| caption_frames | 30.918 |
| sample_fps | 5.064 |
| detect_object_yolo | 7.726 |
| audio_scan | 13.995 |
| asr_timings | 9.596 |
| ast_timings | 21.648 |
| describe_scenes | 6.414 |
| summarize_scenes | 6.425 |
| synthesize_synopsis | 12.561 |
| make_embedding | 2.815 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.644 |
| branch_yolo_total | 12.796 |
| branch_audio_total | 45.248 |
