# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:48:56 UTC | 1A6qJhSsuWc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 236.448 | 0.776 | 79.670 | 33.799 | 20.610 | 21.675 | 5.481 |
| 2026-06-27 14:34:14 UTC | 1A6qJhSsuWc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.053 | 0.800 | 79.333 | 13.443 | 9.222 | 7.631 | 5.992 |

## 2026-06-23 12:48:56 UTC | 1A6qJhSsuWc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1A6qJhSsuWc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `236.448` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 1.495 |
| caption_frames | 58.097 |
| sample_fps | 2.570 |
| detect_object_yolo | 10.903 |
| audio_scan | 14.775 |
| asr_timings | 21.162 |
| ast_timings | 43.725 |
| describe_scenes | 33.799 |
| summarize_scenes | 20.610 |
| synthesize_synopsis | 21.675 |
| make_embedding | 5.481 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.598 |
| branch_yolo_total | 13.479 |
| branch_audio_total | 79.670 |

## 2026-06-27 14:34:14 UTC | 1A6qJhSsuWc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1A6qJhSsuWc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.053` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 1.501 |
| caption_frames | 63.930 |
| sample_fps | 2.602 |
| detect_object_yolo | 11.171 |
| audio_scan | 15.018 |
| asr_timings | 20.131 |
| ast_timings | 44.176 |
| describe_scenes | 13.443 |
| summarize_scenes | 9.222 |
| synthesize_synopsis | 7.631 |
| make_embedding | 5.992 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.437 |
| branch_yolo_total | 13.778 |
| branch_audio_total | 79.333 |
