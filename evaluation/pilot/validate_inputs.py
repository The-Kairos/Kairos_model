"""Measure actual media packet coverage; container duration can be subtitle-only."""
import json
from pathlib import Path
import av

BASE = Path(__file__).resolve().parent
results = {}
for path in (BASE / 'movies').glob('*.mkv'):
    with av.open(str(path)) as container:
        ranges = {}
        for packet in container.demux():
            if packet.stream.type not in ('video', 'audio') or packet.pts is None:
                continue
            key = f'{packet.stream.index}:{packet.stream.type}'
            start = float(packet.pts * packet.time_base)
            end = start + float((packet.duration or 0) * packet.time_base)
            if key not in ranges:
                ranges[key] = {'first_sec': start, 'last_sec': end, 'packets': 0}
            ranges[key]['last_sec'] = max(ranges[key]['last_sec'], end)
            ranges[key]['packets'] += 1
        results[path.name] = {'container_duration_sec': container.duration / av.time_base,
                             'media_packet_ranges': ranges}
        print(path.name, json.dumps(results[path.name]), flush=True)
out = BASE / 'local_20261020/input_checks/packet_ranges.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2), encoding='utf-8')
