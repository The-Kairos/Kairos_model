# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:08:17 UTC | 4VCIae1iyo8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 211.257 | 0.767 | 74.608 | 18.536 | 29.192 | 24.579 | 4.396 |
| 2026-06-24 11:01:49 UTC | 4VCIae1iyo8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.288 | 0.776 | 68.909 | 18.217 | 14.423 | 18.197 | 4.305 |

## 2026-06-23 17:08:17 UTC | 4VCIae1iyo8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4VCIae1iyo8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `211.257` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.156 |
| caption_frames | 44.469 |
| sample_fps | 2.296 |
| detect_object_yolo | 9.890 |
| audio_scan | 15.889 |
| asr_timings | 23.346 |
| ast_timings | 35.365 |
| describe_scenes | 18.536 |
| summarize_scenes | 29.192 |
| synthesize_synopsis | 24.579 |
| make_embedding | 4.396 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.630 |
| branch_yolo_total | 12.192 |
| branch_audio_total | 74.608 |

## 2026-06-24 11:01:49 UTC | 4VCIae1iyo8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4VCIae1iyo8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.288` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 1.175 |
| caption_frames | 46.207 |
| sample_fps | 2.363 |
| detect_object_yolo | 10.260 |
| audio_scan | 16.060 |
| asr_timings | 17.036 |
| ast_timings | 35.804 |
| describe_scenes | 18.217 |
| summarize_scenes | 14.423 |
| synthesize_synopsis | 18.197 |
| make_embedding | 4.305 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.388 |
| branch_yolo_total | 12.629 |
| branch_audio_total | 68.909 |
