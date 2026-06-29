# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:35:04 UTC | 5Ib6GnYyw-o_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.386 | 0.767 | 59.818 | 16.256 | 13.553 | 16.443 | 4.186 |

## 2026-06-24 11:35:04 UTC | 5Ib6GnYyw-o_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5Ib6GnYyw-o_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.386` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.520 |
| caption_frames | 47.873 |
| sample_fps | 2.422 |
| detect_object_yolo | 10.143 |
| audio_scan | 15.923 |
| asr_timings | 9.229 |
| ast_timings | 34.657 |
| describe_scenes | 16.256 |
| summarize_scenes | 13.553 |
| synthesize_synopsis | 16.443 |
| make_embedding | 4.186 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.399 |
| branch_yolo_total | 12.571 |
| branch_audio_total | 59.818 |
