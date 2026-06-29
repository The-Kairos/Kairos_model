# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:14:44 UTC | ljcGQQxxO1E_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 1961.663 | 0.640 | 1869.716 | 21.914 | 8.587 | 17.526 | 3.618 |

## 2026-06-26 17:14:44 UTC | ljcGQQxxO1E_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ljcGQQxxO1E_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1961.663` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 0.827 |
| caption_frames | 23.698 |
| sample_fps | 1.915 |
| detect_object_yolo | 6.842 |
| audio_scan | 10.096 |
| asr_timings | 1831.817 |
| ast_timings | 27.794 |
| describe_scenes | 21.914 |
| summarize_scenes | 8.587 |
| synthesize_synopsis | 17.526 |
| make_embedding | 3.618 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.530 |
| branch_yolo_total | 8.763 |
| branch_audio_total | 1869.716 |
