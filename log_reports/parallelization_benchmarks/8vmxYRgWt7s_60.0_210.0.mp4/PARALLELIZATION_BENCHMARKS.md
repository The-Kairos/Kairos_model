# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:33:36 UTC | 8vmxYRgWt7s_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 176.183 | 0.696 | 55.994 | 13.399 | 10.522 | 33.539 | 3.906 |

## 2026-06-24 17:33:36 UTC | 8vmxYRgWt7s_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8vmxYRgWt7s_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `176.183` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.696 |
| save_clips | - |
| sample_frames | 1.145 |
| caption_frames | 43.966 |
| sample_fps | 2.189 |
| detect_object_yolo | 9.426 |
| audio_scan | 12.801 |
| asr_timings | 10.913 |
| ast_timings | 32.272 |
| describe_scenes | 13.399 |
| summarize_scenes | 10.522 |
| synthesize_synopsis | 33.539 |
| make_embedding | 3.906 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.117 |
| branch_yolo_total | 11.620 |
| branch_audio_total | 55.994 |
