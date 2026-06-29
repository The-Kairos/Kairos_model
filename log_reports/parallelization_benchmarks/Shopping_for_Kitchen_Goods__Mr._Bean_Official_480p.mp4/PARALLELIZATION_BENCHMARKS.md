# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-05-13 07:18:33 UTC | Shopping_for_Kitchen_Goods__Mr._Bean_Official_480p.mp4 | parallel | gemini | gemini-embedding-001 | 89.848 | 0.765 | 19.969 | 22.337 | 25.473 | 16.335 | 0.840 |

## 2026-05-13 07:18:33 UTC | Shopping_for_Kitchen_Goods__Mr._Bean_Official_480p.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/5407106e-daaf-4ce9-a038-2c850d86830d/Shopping_for_Kitchen_Goods__Mr._Bean_Official_480p.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `89.848` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.765 |
| save_clips | - |
| sample_frames | 0.621 |
| caption_frames | 12.560 |
| sample_fps | 3.923 |
| detect_object_yolo | 8.623 |
| audio_scan | 4.678 |
| asr_timings | 0.000 |
| ast_timings | 15.280 |
| describe_scenes | 22.337 |
| summarize_scenes | 25.473 |
| synthesize_synopsis | 16.335 |
| make_embedding | 0.840 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.188 |
| branch_yolo_total | 12.553 |
| branch_audio_total | 19.969 |
