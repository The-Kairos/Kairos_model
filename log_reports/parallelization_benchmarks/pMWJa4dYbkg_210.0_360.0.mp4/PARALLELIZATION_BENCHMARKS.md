# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:01:10 UTC | pMWJa4dYbkg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 78.238 | 0.838 | 32.821 | 4.961 | 3.953 | 12.042 | 1.555 |

## 2026-06-28 08:01:10 UTC | pMWJa4dYbkg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pMWJa4dYbkg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `78.238` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.838 |
| save_clips | - |
| sample_frames | 0.192 |
| caption_frames | 12.245 |
| sample_fps | 1.795 |
| detect_object_yolo | 6.449 |
| audio_scan | 13.687 |
| asr_timings | 11.903 |
| ast_timings | 7.223 |
| describe_scenes | 4.961 |
| summarize_scenes | 3.953 |
| synthesize_synopsis | 12.042 |
| make_embedding | 1.555 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.443 |
| branch_yolo_total | 8.250 |
| branch_audio_total | 32.821 |
