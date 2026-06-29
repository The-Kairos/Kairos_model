# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:40:48 UTC | AEf_3sgs_Ak_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 102.406 | 0.775 | 35.800 | 6.541 | 5.913 | 28.978 | 1.835 |

## 2026-06-24 18:40:48 UTC | AEf_3sgs_Ak_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AEf_3sgs_Ak_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `102.406` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 0.275 |
| caption_frames | 13.258 |
| sample_fps | 1.842 |
| detect_object_yolo | 5.791 |
| audio_scan | 14.953 |
| asr_timings | 10.606 |
| ast_timings | 10.233 |
| describe_scenes | 6.541 |
| summarize_scenes | 5.913 |
| synthesize_synopsis | 28.978 |
| make_embedding | 1.835 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.539 |
| branch_yolo_total | 7.638 |
| branch_audio_total | 35.800 |
