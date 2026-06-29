# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:39:21 UTC | q9IRmUEsz6g_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.739 | 0.675 | 67.057 | 11.868 | 15.008 | 8.492 | 5.087 |

## 2026-06-28 08:39:21 UTC | q9IRmUEsz6g_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q9IRmUEsz6g_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.739` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 1.443 |
| caption_frames | 55.291 |
| sample_fps | 2.361 |
| detect_object_yolo | 11.034 |
| audio_scan | 16.003 |
| asr_timings | 9.945 |
| ast_timings | 41.100 |
| describe_scenes | 11.868 |
| summarize_scenes | 15.008 |
| synthesize_synopsis | 8.492 |
| make_embedding | 5.087 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.740 |
| branch_yolo_total | 13.401 |
| branch_audio_total | 67.057 |
