# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:16:07 UTC | aIGR9knS1B0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.692 | 0.674 | 65.653 | 14.799 | 9.174 | 8.923 | 3.895 |

## 2026-06-26 00:16:07 UTC | aIGR9knS1B0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/aIGR9knS1B0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.692` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 1.239 |
| caption_frames | 42.413 |
| sample_fps | 2.249 |
| detect_object_yolo | 9.250 |
| audio_scan | 5.231 |
| asr_timings | 26.740 |
| ast_timings | 33.674 |
| describe_scenes | 14.799 |
| summarize_scenes | 9.174 |
| synthesize_synopsis | 8.923 |
| make_embedding | 3.895 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.658 |
| branch_yolo_total | 11.504 |
| branch_audio_total | 65.653 |
