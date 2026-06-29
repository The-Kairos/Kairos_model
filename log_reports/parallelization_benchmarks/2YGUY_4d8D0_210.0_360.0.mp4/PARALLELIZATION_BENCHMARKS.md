# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:30:46 UTC | 2YGUY_4d8D0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 231.895 | 0.835 | 65.035 | 29.876 | 29.487 | 22.214 | 5.713 |
| 2026-06-27 15:46:47 UTC | 2YGUY_4d8D0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.657 | 0.836 | 65.888 | 15.577 | 10.235 | 9.478 | 5.754 |

## 2026-06-23 14:30:46 UTC | 2YGUY_4d8D0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2YGUY_4d8D0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `231.895` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.835 |
| save_clips | - |
| sample_frames | 1.771 |
| caption_frames | 61.109 |
| sample_fps | 2.661 |
| detect_object_yolo | 11.781 |
| audio_scan | 6.403 |
| asr_timings | 12.071 |
| ast_timings | 46.554 |
| describe_scenes | 29.876 |
| summarize_scenes | 29.487 |
| synthesize_synopsis | 22.214 |
| make_embedding | 5.713 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 62.886 |
| branch_yolo_total | 14.448 |
| branch_audio_total | 65.035 |

## 2026-06-27 15:46:47 UTC | 2YGUY_4d8D0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2YGUY_4d8D0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.657` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.836 |
| save_clips | - |
| sample_frames | 1.785 |
| caption_frames | 61.223 |
| sample_fps | 2.700 |
| detect_object_yolo | 11.764 |
| audio_scan | 6.499 |
| asr_timings | 12.414 |
| ast_timings | 46.968 |
| describe_scenes | 15.577 |
| summarize_scenes | 10.235 |
| synthesize_synopsis | 9.478 |
| make_embedding | 5.754 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.014 |
| branch_yolo_total | 14.470 |
| branch_audio_total | 65.888 |
