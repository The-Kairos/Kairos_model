# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:43:35 UTC | 38Zc__m3nKQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.900 | 0.653 | 65.600 | 14.447 | 8.755 | 37.502 | 2.040 |
| 2026-06-24 09:39:49 UTC | 38Zc__m3nKQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.806 | 0.676 | 68.038 | 7.377 | 6.261 | 17.212 | 2.026 |

## 2026-06-23 15:43:35 UTC | 38Zc__m3nKQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/38Zc__m3nKQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.900` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.653 |
| save_clips | - |
| sample_frames | 0.444 |
| caption_frames | 17.046 |
| sample_fps | 1.765 |
| detect_object_yolo | 6.258 |
| audio_scan | 12.668 |
| asr_timings | 41.236 |
| ast_timings | 11.688 |
| describe_scenes | 14.447 |
| summarize_scenes | 8.755 |
| synthesize_synopsis | 37.502 |
| make_embedding | 2.040 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.495 |
| branch_yolo_total | 8.029 |
| branch_audio_total | 65.600 |

## 2026-06-24 09:39:49 UTC | 38Zc__m3nKQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/38Zc__m3nKQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.806` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.676 |
| save_clips | - |
| sample_frames | 0.450 |
| caption_frames | 17.212 |
| sample_fps | 1.797 |
| detect_object_yolo | 6.368 |
| audio_scan | 12.811 |
| asr_timings | 43.524 |
| ast_timings | 11.694 |
| describe_scenes | 7.377 |
| summarize_scenes | 6.261 |
| synthesize_synopsis | 17.212 |
| make_embedding | 2.026 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.668 |
| branch_yolo_total | 8.171 |
| branch_audio_total | 68.038 |
