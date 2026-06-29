# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:39:01 UTC | 38Zc__m3nKQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 99.994 | 0.674 | 31.315 | 5.888 | 6.662 | 30.969 | 1.801 |
| 2026-06-24 09:35:48 UTC | 38Zc__m3nKQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.261 | 0.677 | 31.911 | 9.520 | 8.601 | 17.949 | 1.889 |

## 2026-06-23 15:39:01 UTC | 38Zc__m3nKQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/38Zc__m3nKQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `99.994` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 0.323 |
| caption_frames | 13.501 |
| sample_fps | 1.698 |
| detect_object_yolo | 5.815 |
| audio_scan | 12.705 |
| asr_timings | 9.527 |
| ast_timings | 9.075 |
| describe_scenes | 5.888 |
| summarize_scenes | 6.662 |
| synthesize_synopsis | 30.969 |
| make_embedding | 1.801 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.830 |
| branch_yolo_total | 7.519 |
| branch_audio_total | 31.315 |

## 2026-06-24 09:35:48 UTC | 38Zc__m3nKQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/38Zc__m3nKQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.261` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 0.324 |
| caption_frames | 14.320 |
| sample_fps | 1.739 |
| detect_object_yolo | 5.912 |
| audio_scan | 12.881 |
| asr_timings | 9.944 |
| ast_timings | 9.077 |
| describe_scenes | 9.520 |
| summarize_scenes | 8.601 |
| synthesize_synopsis | 17.949 |
| make_embedding | 1.889 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.650 |
| branch_yolo_total | 7.657 |
| branch_audio_total | 31.911 |
