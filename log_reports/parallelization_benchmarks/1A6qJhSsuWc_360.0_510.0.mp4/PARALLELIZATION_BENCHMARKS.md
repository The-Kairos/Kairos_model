# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:52:08 UTC | 1A6qJhSsuWc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 190.997 | 0.766 | 59.884 | 26.253 | 15.487 | 18.863 | 4.558 |
| 2026-06-27 14:37:04 UTC | 1A6qJhSsuWc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.395 | 0.789 | 59.894 | 13.224 | 13.664 | 8.818 | 4.613 |

## 2026-06-23 12:52:08 UTC | 1A6qJhSsuWc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1A6qJhSsuWc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `190.997` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 1.166 |
| caption_frames | 50.397 |
| sample_fps | 2.347 |
| detect_object_yolo | 9.897 |
| audio_scan | 13.664 |
| asr_timings | 8.100 |
| ast_timings | 38.113 |
| describe_scenes | 26.253 |
| summarize_scenes | 15.487 |
| synthesize_synopsis | 18.863 |
| make_embedding | 4.558 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.569 |
| branch_yolo_total | 12.249 |
| branch_audio_total | 59.884 |

## 2026-06-27 14:37:04 UTC | 1A6qJhSsuWc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1A6qJhSsuWc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.395` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.180 |
| caption_frames | 53.216 |
| sample_fps | 2.394 |
| detect_object_yolo | 10.167 |
| audio_scan | 13.903 |
| asr_timings | 7.655 |
| ast_timings | 38.327 |
| describe_scenes | 13.224 |
| summarize_scenes | 13.664 |
| synthesize_synopsis | 8.818 |
| make_embedding | 4.613 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.402 |
| branch_yolo_total | 12.567 |
| branch_audio_total | 59.894 |
