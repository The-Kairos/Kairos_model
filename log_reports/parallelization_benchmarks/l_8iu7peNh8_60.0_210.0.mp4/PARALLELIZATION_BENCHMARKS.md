# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:29:44 UTC | l_8iu7peNh8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.389 | 0.669 | 69.233 | 12.878 | 21.147 | 18.489 | 3.035 |

## 2026-06-26 15:29:44 UTC | l_8iu7peNh8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l_8iu7peNh8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.389` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.669 |
| save_clips | - |
| sample_frames | 0.864 |
| caption_frames | 31.804 |
| sample_fps | 2.009 |
| detect_object_yolo | 7.821 |
| audio_scan | 10.838 |
| asr_timings | 33.610 |
| ast_timings | 24.776 |
| describe_scenes | 12.878 |
| summarize_scenes | 21.147 |
| synthesize_synopsis | 18.489 |
| make_embedding | 3.035 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.673 |
| branch_yolo_total | 9.835 |
| branch_audio_total | 69.233 |
