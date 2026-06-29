# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:30:39 UTC | ccMqbhacbpY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 145.299 | 0.820 | 52.272 | 8.422 | 9.532 | 22.655 | 3.279 |

## 2026-06-26 02:30:39 UTC | ccMqbhacbpY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ccMqbhacbpY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.299` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 1.218 |
| caption_frames | 35.221 |
| sample_fps | 2.278 |
| detect_object_yolo | 8.173 |
| audio_scan | 14.067 |
| asr_timings | 11.559 |
| ast_timings | 26.637 |
| describe_scenes | 8.422 |
| summarize_scenes | 9.532 |
| synthesize_synopsis | 22.655 |
| make_embedding | 3.279 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.444 |
| branch_yolo_total | 10.456 |
| branch_audio_total | 52.272 |
