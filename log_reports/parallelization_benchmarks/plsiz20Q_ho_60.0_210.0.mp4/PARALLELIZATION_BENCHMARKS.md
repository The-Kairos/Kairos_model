# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:17:16 UTC | plsiz20Q_ho_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 146.331 | 0.660 | 50.327 | 11.929 | 17.128 | 11.833 | 3.244 |

## 2026-06-28 08:17:16 UTC | plsiz20Q_ho_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/plsiz20Q_ho_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `146.331` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.660 |
| save_clips | - |
| sample_frames | 1.227 |
| caption_frames | 37.535 |
| sample_fps | 2.181 |
| detect_object_yolo | 8.830 |
| audio_scan | 12.902 |
| asr_timings | 10.284 |
| ast_timings | 27.133 |
| describe_scenes | 11.929 |
| summarize_scenes | 17.128 |
| synthesize_synopsis | 11.833 |
| make_embedding | 3.244 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.768 |
| branch_yolo_total | 11.016 |
| branch_audio_total | 50.327 |
