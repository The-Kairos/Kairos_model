# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:22:42 UTC | Jhr9DdFk0cI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.499 | 0.818 | 103.970 | 12.383 | 21.195 | 18.769 | 2.754 |

## 2026-06-25 05:22:42 UTC | Jhr9DdFk0cI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Jhr9DdFk0cI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.499` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 0.899 |
| caption_frames | 27.891 |
| sample_fps | 2.144 |
| detect_object_yolo | 7.271 |
| audio_scan | 10.605 |
| asr_timings | 72.358 |
| ast_timings | 20.998 |
| describe_scenes | 12.383 |
| summarize_scenes | 21.195 |
| synthesize_synopsis | 18.769 |
| make_embedding | 2.754 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.795 |
| branch_yolo_total | 9.421 |
| branch_audio_total | 103.970 |
