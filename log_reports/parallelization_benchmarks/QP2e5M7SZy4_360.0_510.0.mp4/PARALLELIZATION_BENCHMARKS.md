# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:05:33 UTC | QP2e5M7SZy4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.040 | 0.795 | 46.125 | 20.915 | 10.938 | 33.350 | 2.853 |

## 2026-06-25 15:05:33 UTC | QP2e5M7SZy4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QP2e5M7SZy4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.040` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 0.735 |
| caption_frames | 30.291 |
| sample_fps | 2.103 |
| detect_object_yolo | 7.522 |
| audio_scan | 15.627 |
| asr_timings | 9.257 |
| ast_timings | 21.233 |
| describe_scenes | 20.915 |
| summarize_scenes | 10.938 |
| synthesize_synopsis | 33.350 |
| make_embedding | 2.853 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.032 |
| branch_yolo_total | 9.631 |
| branch_audio_total | 46.125 |
