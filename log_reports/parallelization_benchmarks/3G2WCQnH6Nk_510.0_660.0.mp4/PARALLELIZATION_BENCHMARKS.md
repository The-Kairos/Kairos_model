# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:51:37 UTC | 3G2WCQnH6Nk_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.846 | 0.775 | 66.647 | 16.232 | 14.276 | 20.125 | 2.833 |
| 2026-06-24 09:48:12 UTC | 3G2WCQnH6Nk_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.902 | 0.767 | 65.681 | 10.200 | 29.876 | 22.019 | 2.830 |

## 2026-06-23 15:51:37 UTC | 3G2WCQnH6Nk_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.846` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 0.818 |
| caption_frames | 27.192 |
| sample_fps | 2.083 |
| detect_object_yolo | 7.452 |
| audio_scan | 14.832 |
| asr_timings | 31.087 |
| ast_timings | 20.720 |
| describe_scenes | 16.232 |
| summarize_scenes | 14.276 |
| synthesize_synopsis | 20.125 |
| make_embedding | 2.833 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.016 |
| branch_yolo_total | 9.541 |
| branch_audio_total | 66.647 |

## 2026-06-24 09:48:12 UTC | 3G2WCQnH6Nk_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.902` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 0.819 |
| caption_frames | 27.754 |
| sample_fps | 2.107 |
| detect_object_yolo | 7.464 |
| audio_scan | 14.916 |
| asr_timings | 30.133 |
| ast_timings | 20.622 |
| describe_scenes | 10.200 |
| summarize_scenes | 29.876 |
| synthesize_synopsis | 22.019 |
| make_embedding | 2.830 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.578 |
| branch_yolo_total | 9.577 |
| branch_audio_total | 65.681 |
