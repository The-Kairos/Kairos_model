# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:46:17 UTC | 48I5xM9Yq-4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.127 | 0.638 | 58.356 | 19.642 | 21.751 | 21.977 | 4.216 |
| 2026-06-24 10:41:22 UTC | 48I5xM9Yq-4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.852 | 0.640 | 57.501 | 18.316 | 14.703 | 29.162 | 4.193 |

## 2026-06-23 16:46:17 UTC | 48I5xM9Yq-4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/48I5xM9Yq-4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.127` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.638 |
| save_clips | - |
| sample_frames | 1.549 |
| caption_frames | 45.016 |
| sample_fps | 2.307 |
| detect_object_yolo | 9.282 |
| audio_scan | 14.852 |
| asr_timings | 7.818 |
| ast_timings | 35.678 |
| describe_scenes | 19.642 |
| summarize_scenes | 21.751 |
| synthesize_synopsis | 21.977 |
| make_embedding | 4.216 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.570 |
| branch_yolo_total | 11.595 |
| branch_audio_total | 58.356 |

## 2026-06-24 10:41:22 UTC | 48I5xM9Yq-4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/48I5xM9Yq-4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.852` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 1.579 |
| caption_frames | 44.815 |
| sample_fps | 2.307 |
| detect_object_yolo | 9.227 |
| audio_scan | 14.895 |
| asr_timings | 7.011 |
| ast_timings | 35.587 |
| describe_scenes | 18.316 |
| summarize_scenes | 14.703 |
| synthesize_synopsis | 29.162 |
| make_embedding | 4.193 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.400 |
| branch_yolo_total | 11.540 |
| branch_audio_total | 57.501 |
