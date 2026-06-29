# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:57:45 UTC | 0iz-ty5Wl3U_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.524 | 0.663 | 53.176 | 9.190 | 9.673 | 11.657 | 3.346 |

## 2026-06-27 13:57:45 UTC | 0iz-ty5Wl3U_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0iz-ty5Wl3U_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.524` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.663 |
| save_clips | - |
| sample_frames | 1.059 |
| caption_frames | 38.041 |
| sample_fps | 2.110 |
| detect_object_yolo | 8.210 |
| audio_scan | 13.887 |
| asr_timings | 12.101 |
| ast_timings | 27.180 |
| describe_scenes | 9.190 |
| summarize_scenes | 9.673 |
| synthesize_synopsis | 11.657 |
| make_embedding | 3.346 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.107 |
| branch_yolo_total | 10.326 |
| branch_audio_total | 53.176 |
