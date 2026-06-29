# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:15:57 UTC | _Jq5O0D8q8M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 176.760 | 0.711 | 69.917 | 13.409 | 12.620 | 12.056 | 4.204 |

## 2026-06-25 23:15:57 UTC | _Jq5O0D8q8M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_Jq5O0D8q8M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `176.760` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.711 |
| save_clips | - |
| sample_frames | 1.581 |
| caption_frames | 48.298 |
| sample_fps | 2.395 |
| detect_object_yolo | 10.076 |
| audio_scan | 12.936 |
| asr_timings | 20.422 |
| ast_timings | 36.551 |
| describe_scenes | 13.409 |
| summarize_scenes | 12.620 |
| synthesize_synopsis | 12.056 |
| make_embedding | 4.204 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.885 |
| branch_yolo_total | 12.477 |
| branch_audio_total | 69.917 |
