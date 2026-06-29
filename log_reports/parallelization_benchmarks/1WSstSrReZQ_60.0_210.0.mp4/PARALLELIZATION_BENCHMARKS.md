# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:46:41 UTC | 1WSstSrReZQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 231.285 | 0.779 | 59.865 | 44.493 | 20.987 | 29.638 | 5.083 |
| 2026-06-27 15:16:15 UTC | 1WSstSrReZQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.506 | 0.802 | 60.435 | 15.224 | 10.422 | 9.844 | 5.113 |

## 2026-06-23 13:46:41 UTC | 1WSstSrReZQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1WSstSrReZQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `231.285` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.404 |
| caption_frames | 54.580 |
| sample_fps | 2.512 |
| detect_object_yolo | 10.567 |
| audio_scan | 9.561 |
| asr_timings | 9.040 |
| ast_timings | 41.257 |
| describe_scenes | 44.493 |
| summarize_scenes | 20.987 |
| synthesize_synopsis | 29.638 |
| make_embedding | 5.083 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.990 |
| branch_yolo_total | 13.085 |
| branch_audio_total | 59.865 |

## 2026-06-27 15:16:15 UTC | 1WSstSrReZQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1WSstSrReZQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.506` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 1.425 |
| caption_frames | 56.271 |
| sample_fps | 2.551 |
| detect_object_yolo | 10.973 |
| audio_scan | 9.644 |
| asr_timings | 8.877 |
| ast_timings | 41.905 |
| describe_scenes | 15.224 |
| summarize_scenes | 10.422 |
| synthesize_synopsis | 9.844 |
| make_embedding | 5.113 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.702 |
| branch_yolo_total | 13.530 |
| branch_audio_total | 60.435 |
