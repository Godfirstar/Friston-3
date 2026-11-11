"""SO-101 示教数据：读取、视频定位、统计和归一化。

trainer.py 通过 sample_data 取得动作窗口，通过 sample_subtask_frame 取得高层语义监督和当前观测。
数据阅读顺序：load_rows → training_stats → video_locator/decode_frame → normalize。
state/action 的六个槽位使用同一 JOINTS 顺序，但分别使用各自的统计量。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ACTION, DATASET_JOINTS as JOINTS, STATE
from ..models.weights import fingerprint, read_json

TASK = 'Pick up the cube and place it in the marked area.'


def write_json(path: Path, data: Any) -> None:
    # 【调用方】trainer.save_checkpoint；把Python对象写成JSON；-> None表示不返回业务值。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def event(label: str, data: Any) -> None:
    print(label, json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


def field(row: dict, name: str) -> Any:
    """Accept flattened v3 metadata; also accept genuinely nested equivalents."""
    if name in row:
        return row[name]
    value = row
    for part in name.split('/'):
        value = value[part]
    return value


def names(feature: dict) -> list[str]:
    result = feature.get('names')
    if isinstance(result, dict):
        result = result.get('motors')
    if not isinstance(result, list):
        raise ValueError('State/action features must declare joint names')
    return [str(name).removesuffix('.pos') for name in result]


def apply_subtask_segments(groups: dict[int, list[dict]], segments: list[dict]) -> None:
    """把可选人工标注对齐到原始帧。显式 null 是可见的完成态，未标注的 episode 仍可训练旧动作策略。"""
    if not isinstance(segments, list) or not segments:
        raise ValueError('meta/subtasks.json must contain a nonempty list of annotated segments')
    by_episode = {}
    for segment in segments:
        if not isinstance(segment, dict) or not {'episode_index', 'start_frame', 'stop_frame',
                                                  'global_goal', 'subtask'} <= segment.keys():
            raise ValueError('Each subtask segment needs episode_index, start_frame, stop_frame, global_goal, subtask')
        ep, start, stop = (segment[k] for k in ('episode_index', 'start_frame', 'stop_frame'))
        if any(type(v) is not int for v in (ep, start, stop)) or ep not in groups or not 0 <= start < stop <= len(groups[ep]):
            raise ValueError(f'Invalid subtask segment bounds {ep=}, {start=}, {stop=}')
        goal, subtask = segment['global_goal'], segment['subtask']
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError('Each annotated segment needs a nonempty global_goal')
        if subtask is not None and (not isinstance(subtask, str) or not subtask.strip()):
            raise ValueError('subtask must be a nonempty action instruction or null for observed completion')
        by_episode.setdefault(ep, []).append((start, stop, goal.strip(), subtask.strip() if subtask else None))
    for ep, spans in by_episode.items():
        expected, global_goal = 0, spans[0][2]
        for start, stop, goal, subtask in sorted(spans, key=lambda span: span[0]):
            if start != expected or goal != global_goal:
                raise ValueError(f'Episode {ep}: subtask ranges must partition one global goal without gaps or overlap')
            for row in groups[ep][start:stop]:
                row.update(global_goal=goal, subtask=subtask)
            expected = stop
        if expected != len(groups[ep]):
            raise ValueError(f'Episode {ep}: subtask annotation stops before the last frame')


def apply_grounding_frames(groups, entries, image_keys):
    """将相关物体框对齐到原始单帧。坐标为对应相机原图归一化 xyxy；空列表是显式负例。"""
    if not isinstance(entries, list):
        raise ValueError("meta/grounding.json must be a list of frame annotations")
    for item in entries:
        if not isinstance(item, dict) or not {"episode_index", "frame_index", "boxes"} <= item.keys():
            raise ValueError("Grounding entries need episode_index, frame_index and boxes")
        ep, frame, boxes = item["episode_index"], item["frame_index"], item["boxes"]
        if (type(ep) is not int or type(frame) is not int or ep not in groups or
                not 0 <= frame < len(groups[ep]) or groups[ep][frame]["frame_index"] != frame):
            raise ValueError(f"Unknown grounding frame {ep=}, {frame=}")
        row = groups[ep][frame]
        if "boxes" in row or not isinstance(boxes, list):
            raise ValueError(f"Duplicate grounding annotation or invalid boxes at {ep=}, {frame=}")
        validated = []
        for box in boxes:
            if not isinstance(box, dict) or not isinstance(box.get("view"), str) or box["view"] not in image_keys:
                raise ValueError(f"Grounding box needs a dataset camera view at {ep=}, {frame=}")
            coords = box.get("xyxy")
            if (not isinstance(coords, list) or len(coords) != 4 or
                    any(type(v) not in (int, float) for v in coords) or
                    not (0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1)):
                raise ValueError(f"Expected normalized [xmin,ymin,xmax,ymax] in original view at {ep=}, {frame=}")
            validated.append({"view": box["view"], "xyxy": list(map(float, coords))})
        row["boxes"] = validated


def load_rows(root: Path) -> tuple[dict, dict[int, list[dict]], dict[int, dict]]:
    """Read local scalar/action Parquet and v3 episode locators, not video pixels."""
    # 【调用方】trainer.run；root是数据集目录，返回(info,groups,episodes)三个对象。
    # tuple[...]声明返回元组；dict[int,list[dict]]表示episode整数键对应逐帧字典列表。
    import pyarrow.parquet as pq
    # 函数内导入：只有读取数据时才需要pyarrow；pq是模块别名，不是一个已打开文件。
    info = read_json(root / 'meta' / 'info.json')
    for key in (STATE, ACTION):
        if names(info['features'][key]) != JOINTS:
            raise ValueError(f'{key} joint order does not match {JOINTS}')
    groups: dict[int, list[dict]] = {}
    # 变量注解说明容器结构；{}实际创建空字典，后面逐帧append。
    task_path = root / 'meta' / 'tasks.parquet'
    tasks = {}
    if task_path.is_file():
        for item in pq.read_table(task_path).to_pylist():
            # read_table读Parquet表；to_pylist转换为[每行dict,...]，便于按键取值。
            task_id, task = int(item['task_index']), item['task']
            if task_id in tasks or not isinstance(task, str) or not task.strip():
                raise ValueError('Dataset task labels must be nonempty and uniquely indexed')
            tasks[task_id] = task.strip()
    info = dict(info, tasks=tasks)
    # 建新字典：复制原info键值并新增/覆盖tasks键，不是原地更改输入文件。
    columns = [STATE, ACTION, 'episode_index', 'frame_index', 'timestamp', 'task_index']
    for path in sorted((root / 'data').rglob('*.parquet')):
        # Path的/拼路径；rglob递归寻找匹配文件；sorted让处理顺序可重复。
        available = pq.read_schema(path).names
        selected = [name for name in columns if name in available]
        # 列表推导式仅保留文件里确实存在的列，兼容旧数据缺少task_index的情况。
        for row in pq.read_table(path, columns=selected).to_pylist():
            task_id = row.get('task_index')
            if tasks:
                if task_id is None and len(tasks) == 1:
                    task_id = next(iter(tasks))
                if task_id not in tasks:
                    raise ValueError(f'Unknown task_index {task_id} in {path}')
                row['task'] = tasks[int(task_id)]
            groups.setdefault(int(row['episode_index']), []).append(row)
            # setdefault：键不存在时放入新空list；随后append把当前行追加到该段list末尾。
    if not groups:
        raise ValueError(f'No recorded Parquet rows found in {root / "data"}')
    for episode, rows in groups.items():
        rows.sort(key=lambda row: int(row['frame_index']))
        # list.sort原地排序；key接收函数；lambda把每帧字典转换成排序所用整数下标。
        if [int(row['frame_index']) for row in rows] != list(range(len(rows))):
            raise ValueError(f'Episode {episode}: duplicate or noncontiguous frame_index')
        timestamps = np.asarray([row['timestamp'] for row in rows], dtype=float)
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f'Episode {episode}: timestamps must be finite and increasing')
        for key in (STATE, ACTION):
            array = np.asarray([row[key] for row in rows], dtype=np.float32)
            if array.shape != (len(rows), 6) or not np.isfinite(array).all():
                raise ValueError(f'Episode {episode}: invalid {key}, shape {array.shape}')
    if sum(map(len, groups.values())) != int(info['total_frames']):
        raise ValueError('Parquet row count differs from meta/info.json total_frames')
    episodes: dict[int, dict] = {}
    for path in sorted((root / 'meta' / 'episodes').rglob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            eid = int(row['episode_index'])
            if eid in episodes:
                raise ValueError(f'Duplicate episode locator for {eid}')
            episodes[eid] = row
    if set(episodes) != set(groups):
        raise ValueError('This reader requires finalized v3 meta/episodes/*.parquet for every episode')
    for eid, rows in groups.items():
        if 'length' in episodes[eid] and int(episodes[eid]['length']) != len(rows):
            raise ValueError(f'Episode {eid}: metadata length differs from saved rows')
    annotations = root / 'meta' / 'subtasks.json'
    if annotations.is_file():
        apply_subtask_segments(groups, read_json(annotations))
        info['subtasks_sha256'] = fingerprint(annotations)
    grounding = root / 'meta' / 'grounding.json'
    if grounding.is_file():
        views = [key for key in info['features'] if key.startswith('observation.images.')]
        apply_grounding_frames(groups, read_json(grounding), views)
        info['grounding_sha256'] = fingerprint(grounding)
    return info, groups, episodes


def action_label(row: dict, fallback: str | None) -> str | None:
    if 'subtask' in row:
        return row['subtask']
    label = row.get('task') or fallback
    if not label:
        raise ValueError('Missing legacy task label; provide --task')
    return label


def training_stats(groups: dict[int, list[dict]], train_ids: list[int], *, require_nonzero=True) -> dict:
    """Statistics use training episodes ONLY; never the whole-dataset sidecar."""
    # 【调用方】trainer.run；输入分组帧与训练段ID，返回state/action各自q01/q99/count字典。
    # 参数中的*表示之后的require_nonzero必须按名称传，不可含糊地当作第三个位置参数。
    stats = {}
    for key in (STATE, ACTION):
        array = np.asarray([row[key] for eid in train_ids for row in groups[eid]], dtype=np.float64)
        # 两层for的列表推导：先遍历段，再遍历段内帧；np.asarray将值整理成[N,6]数组。
        # [训练帧数, 6] 沿帧轴统计，六个关节各有自己的尺度；验证集不参与拟合。
        low, high = np.quantile(array, [0.01, 0.99], axis=0, method='linear')
        # axis=0沿帧轴求分位数，每个关节单独统计；返回[2,6]，解包成两个[6]数组。
        if require_nonzero and np.any(high.astype(np.float32) <= low.astype(np.float32)):
            raise ValueError(f'{key}: zero/inverted quantile span; no bounds will be fabricated')
        stats[key] = {'q01': low.tolist(), 'q99': high.tolist(), 'count': len(array)}
    return stats


def normalize(value: Any, stats: dict) -> np.ndarray:
    # 【调用方】sample_data/评估指标；value可为[6]或[H,6]，返回同shape float32数组。
    # sample_data 的状态/动作都经过这里。q01→-1、q99→1，超出分位数仍保留数值。
    # 这是数据坐标变换，不是关节限位；推理必须用同一份 stats 做逆变换。
    array = np.asarray(value, dtype=np.float32)
    lo, hi = np.asarray(stats['q01'], dtype=np.float32), np.asarray(stats['q99'], dtype=np.float32)
    # lo/hi各为[6]；NumPy广播会把它们应用到每个时间行，无需手写逐帧for。
    return 2 * (array - lo) / (hi - lo) - 1  # No clipping.


def video_locator(root: Path, info: dict, meta: dict, key: str, row: dict) -> tuple[Path, float]:
    # 【调用方】sample_images；根据episode元数据返回(视频Path,文件内目标秒数)。
    # meta是该episode定位信息；row是当前采样帧；视频文件号不等于episode号。
    prefix = f'videos/{key}'
    chunk = int(field(meta, prefix + '/chunk_index'))
    file_index = int(field(meta, prefix + '/file_index'))
    start = float(field(meta, prefix + '/from_timestamp'))
    target = start + float(row['timestamp'])
    template = info.get('video_path') or 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'
    relative = template.format(video_key=key, chunk_index=chunk, file_index=file_index,
                               episode_index=int(row['episode_index']))
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, target


def decode_frame(path: Path, timestamp: float, fps: float) -> dict:
    """Find the nearest presentation timestamp; do not equate episode and file IDs."""
    # 【调用方】sample_images；返回rgb[高,宽,3]与定位误差字典；只解码所需附近帧。
    import av
    with av.open(str(path)) as container:
        # PyAV的open返回媒体容器；with负责退出时关闭；as把该对象绑定为container。
        stream = container.streams.video[0]
        container.seek(max(0, int(timestamp / float(stream.time_base))),
                       stream=stream, backward=True, any_frame=False)
        best = None
        best_time = None
        distance = float('inf')
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = float(frame.pts * frame.time_base)
            if abs(t - timestamp) < distance:
                best, best_time, distance = frame, t, abs(t - timestamp)
            if t >= timestamp:
                break
        if best is None:
            raise RuntimeError(f'No decodable frame near {timestamp} in {path}')
        # A time-index integrity check, not a robot motion threshold.
        if distance > 0.5 / fps + 1e-3:
            raise ValueError(f'Video/row mismatch: wanted {timestamp}s, nearest frame {best_time}s')
        rgb = best.to_ndarray(format='rgb24')
    return {'path': str(path), 'requested_time_s': timestamp, 'decoded_time_s': best_time,
            'absolute_difference_s': distance, 'rgb': rgb,
            'hardware_synchronization_verified': False}


def sample_data(root, info, groups, episode_meta, stats, args, episode, start):
    # 【调用关系】training_prefix / evaluate → 本函数 → data 的视频定位、解码、归一化。
    # 输入 root 是数据集 Path；info/groups/episode_meta 来自 data.load_rows；stats 是训练统计。
    # episode 是段编号，start 是这段已选行列表中的起点下标；args.horizon 是动作块长度。
    # 返回五元组：state[6]、actions[H,6]、图像字典、视频定位字典、任务字符串。
    # 一份样本 = 起点的图像/状态/任务 + 从起点开始连续 H 帧的示教 action。
    # future action 是监督标签，不是模型的条件输入；窗口不能跨 episode 或任务标签。
    rows = groups[episode]
    # groups 是 {episode编号: [每帧字典,...]}；[] 按键取出这一段的全部已选帧。
    horizon = args.horizon
    # len(rows) 是帧数；start+horizon 是切片右端点，右端点可等于长度但不含该位置。
    if start < 0 or start + horizon > len(rows):
        raise ValueError("Action chunk crosses an episode boundary")
    row = rows[start]
    # row 是起点帧的 dict，后续图像、状态都取这一时刻，不读取未来图像作为条件。
    task = action_label(row, args.task)
    if task is None or any(action_label(item, args.task) != task for item in rows[start:start + horizon]):
        raise ValueError("Action chunk includes a completed goal or crosses a subtask boundary")
    images, videos = sample_images(root, info, episode_meta[episode], row, args.image_keys)
    state = normalize(row[STATE], stats[STATE])
    actions = normalize([r[ACTION] for r in rows[start:start + horizon]], stats[ACTION])
    return state, actions, images, videos, task


def sample_subtask_frame(root, info, groups, episode_meta, stats, args, episode, start):
    """取一个已标注观测学习 p(子任务 | 图像、状态、总目标)，无需未来的 H 帧。"""
    row = groups[episode][start]
    # 没有标注和人工确认已经完成是两种不同的监督条件。后者保留显式 None。
    if "global_goal" not in row or "subtask" not in row:
        raise ValueError(f"Missing subtask annotation for episode {episode}, frame {row['frame_index']}")
    views = getattr(args, "semantic_image_keys", None) or args.image_keys
    images, _ = sample_images(root, info, episode_meta[episode], row, views)
    state = normalize(row[STATE], stats[STATE])
    return state, images, row["global_goal"], row["subtask"]


def sample_images(root, info, episode_meta, row, image_keys):
    """高低层共用同一起点帧的多相机图像定位，不读取未来图像。"""
    images, videos = {}, {}
    # 逗号左边接收右边两个新字典；不是让 images 和 videos 共用同一字典。
    for key in image_keys:
        # 每个 key 类似 observation.images.base_0_rgb；多视角分别定位同一时刻的图像。
        video_path, video_time = video_locator(root, info, episode_meta, key, row)
        # video_locator 返回 (Path, 秒数)；这里用元组解包把两个结果分别赋值。
        decoded = decode_frame(video_path, video_time, float(info["fps"]))
        # float 把采样频率转成浮点数；decode_frame 返回带 rgb 数组及时间信息的字典。
        images[key] = decoded.pop("rgb")
        # pop 取出并删除 rgb 键：像素进入 images，剩余 decoded 只保留定位/误差信息。
        videos[key] = decoded
    return images, videos
