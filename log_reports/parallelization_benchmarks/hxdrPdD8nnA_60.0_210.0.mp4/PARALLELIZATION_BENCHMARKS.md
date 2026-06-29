# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:28:31 UTC | hxdrPdD8nnA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 201.081 | 0.797 | 61.031 | 18.674 | 24.600 | 27.520 | 4.204 |

## 2026-06-26 07:28:31 UTC | hxdrPdD8nnA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hxdrPdD8nnA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `201.081` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 1.165 |
| caption_frames | 49.206 |
| sample_fps | 2.336 |
| detect_object_yolo | 10.103 |
| audio_scan | 16.164 |
| asr_timings | 9.484 |
| ast_timings | 35.374 |
| describe_scenes | 18.674 |
| summarize_scenes | 24.600 |
| synthesize_synopsis | 27.520 |
| make_embedding | 4.204 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.378 |
| branch_yolo_total | 12.445 |
| branch_audio_total | 61.031 |
