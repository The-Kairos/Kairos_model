# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:46:29 UTC | Kd6pVAb_tHs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 127.874 | 0.817 | 41.228 | 18.999 | 8.783 | 11.042 | 3.387 |

## 2026-06-25 06:46:29 UTC | Kd6pVAb_tHs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Kd6pVAb_tHs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.874` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.817 |
| save_clips | - |
| sample_frames | 1.282 |
| caption_frames | 39.940 |
| sample_fps | 2.319 |
| detect_object_yolo | 8.840 |
| audio_scan | 3.843 |
| asr_timings | 0.000 |
| ast_timings | 27.194 |
| describe_scenes | 18.999 |
| summarize_scenes | 8.783 |
| synthesize_synopsis | 11.042 |
| make_embedding | 3.387 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.228 |
| branch_yolo_total | 11.166 |
| branch_audio_total | 31.046 |
