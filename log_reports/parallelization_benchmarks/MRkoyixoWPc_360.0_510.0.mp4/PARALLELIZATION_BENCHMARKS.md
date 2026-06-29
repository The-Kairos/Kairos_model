# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:40:28 UTC | MRkoyixoWPc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.484 | 0.680 | 40.285 | 10.240 | 8.847 | 27.796 | 2.553 |

## 2026-06-25 09:40:28 UTC | MRkoyixoWPc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MRkoyixoWPc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.484` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.680 |
| save_clips | - |
| sample_frames | 0.633 |
| caption_frames | 25.329 |
| sample_fps | 1.898 |
| detect_object_yolo | 6.825 |
| audio_scan | 10.684 |
| asr_timings | 11.027 |
| ast_timings | 18.566 |
| describe_scenes | 10.240 |
| summarize_scenes | 8.847 |
| synthesize_synopsis | 27.796 |
| make_embedding | 2.553 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.968 |
| branch_yolo_total | 8.728 |
| branch_audio_total | 40.285 |
