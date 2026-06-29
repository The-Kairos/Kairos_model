# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-10 19:03:46 UTC | IMG_5672.mov | parallel | gemini | gemini-embedding-001 | 37.015 | 0.551 | 6.787 | 8.835 | 6.051 | 9.714 | 0.682 |

## 2026-04-10 19:03:46 UTC | IMG_5672.mov | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/fa81395a-184d-4219-80c2-95c004a8aeb2/IMG_5672.mov`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `37.015` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.551 |
| save_clips | - |
| sample_frames | 1.015 |
| caption_frames | 2.205 |
| sample_fps | 0.894 |
| detect_object_yolo | 1.021 |
| audio_scan | 3.911 |
| asr_timings | 2.754 |
| ast_timings | 2.869 |
| describe_scenes | 8.835 |
| summarize_scenes | 6.051 |
| synthesize_synopsis | 9.714 |
| make_embedding | 0.682 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 3.229 |
| branch_yolo_total | 1.923 |
| branch_audio_total | 6.787 |
