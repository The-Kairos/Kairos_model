# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:03:34 UTC | c8EAPVV4dVc_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.233 | 0.814 | 58.054 | 13.864 | 17.053 | 15.879 | 3.324 |

## 2026-06-26 02:03:34 UTC | c8EAPVV4dVc_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/c8EAPVV4dVc_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.233` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 0.971 |
| caption_frames | 37.110 |
| sample_fps | 2.204 |
| detect_object_yolo | 8.564 |
| audio_scan | 16.238 |
| asr_timings | 14.124 |
| ast_timings | 27.684 |
| describe_scenes | 13.864 |
| summarize_scenes | 17.053 |
| synthesize_synopsis | 15.879 |
| make_embedding | 3.324 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.087 |
| branch_yolo_total | 10.773 |
| branch_audio_total | 58.054 |
