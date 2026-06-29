# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:07:56 UTC | UWS6H8snDgA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.892 | 0.674 | 78.399 | 15.659 | 21.258 | 18.880 | 3.562 |

## 2026-06-25 19:07:56 UTC | UWS6H8snDgA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UWS6H8snDgA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.892` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 1.011 |
| caption_frames | 41.140 |
| sample_fps | 2.040 |
| detect_object_yolo | 8.878 |
| audio_scan | 16.051 |
| asr_timings | 32.321 |
| ast_timings | 30.019 |
| describe_scenes | 15.659 |
| summarize_scenes | 21.258 |
| synthesize_synopsis | 18.880 |
| make_embedding | 3.562 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.157 |
| branch_yolo_total | 10.923 |
| branch_audio_total | 78.399 |
