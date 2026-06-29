# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:59:50 UTC | bUa-0ptWL5M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.960 | 0.782 | 43.865 | 8.464 | 12.178 | 7.593 | 2.554 |

## 2026-06-26 00:59:50 UTC | bUa-0ptWL5M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bUa-0ptWL5M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.960` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.745 |
| caption_frames | 26.452 |
| sample_fps | 2.060 |
| detect_object_yolo | 6.878 |
| audio_scan | 14.864 |
| asr_timings | 10.027 |
| ast_timings | 18.966 |
| describe_scenes | 8.464 |
| summarize_scenes | 12.178 |
| synthesize_synopsis | 7.593 |
| make_embedding | 2.554 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.203 |
| branch_yolo_total | 8.944 |
| branch_audio_total | 43.865 |
