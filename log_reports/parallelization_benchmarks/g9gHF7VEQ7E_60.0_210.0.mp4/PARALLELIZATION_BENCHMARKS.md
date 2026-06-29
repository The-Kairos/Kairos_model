# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:07:35 UTC | g9gHF7VEQ7E_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 200.167 | 0.674 | 66.762 | 18.035 | 22.242 | 10.902 | 5.739 |

## 2026-06-26 05:07:35 UTC | g9gHF7VEQ7E_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/g9gHF7VEQ7E_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `200.167` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 1.521 |
| caption_frames | 59.248 |
| sample_fps | 2.457 |
| detect_object_yolo | 11.180 |
| audio_scan | 10.877 |
| asr_timings | 8.957 |
| ast_timings | 46.920 |
| describe_scenes | 18.035 |
| summarize_scenes | 22.242 |
| synthesize_synopsis | 10.902 |
| make_embedding | 5.739 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.775 |
| branch_yolo_total | 13.643 |
| branch_audio_total | 66.762 |
