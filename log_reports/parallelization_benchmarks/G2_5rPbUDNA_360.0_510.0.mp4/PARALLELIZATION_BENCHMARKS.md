# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:13:48 UTC | G2_5rPbUDNA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 272.681 | 0.642 | 83.673 | 29.281 | 38.454 | 13.708 | 6.709 |

## 2026-06-25 01:13:48 UTC | G2_5rPbUDNA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G2_5rPbUDNA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `272.681` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.642 |
| save_clips | - |
| sample_frames | 1.783 |
| caption_frames | 81.884 |
| sample_fps | 2.538 |
| detect_object_yolo | 14.458 |
| audio_scan | 11.693 |
| asr_timings | 10.808 |
| ast_timings | 59.258 |
| describe_scenes | 29.281 |
| summarize_scenes | 38.454 |
| synthesize_synopsis | 13.708 |
| make_embedding | 6.709 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 83.673 |
| branch_yolo_total | 17.002 |
| branch_audio_total | 81.768 |
