#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZCD MVP: language/JSON -> skills -> real executor -> encoder feedback.

python -m friston3.arms.taught.task --demo --execute
python -m friston3.arms.taught.task --instruction "先回参考位，沿基座X正方向移动5厘米，再回参考位" --execute

--execute is the only write authorization. Without it the task is preplanned
from live encoders with NO writes. --plan-only does not open a serial port.
Only one process owns COM9: close teleoperation and the existing arms service.
This is an empty-gripper reach/composition experiment, NOT physical pick/place.
No camera verification, force control, contact learning or collision model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import URLError, HTTPError

from . import move as motion

SKILLS = ("move_delta", "home", "return_start", "wait")
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "reason", "steps"],
    "properties": {
        "status": {"type": "string", "enum": ["plan", "unsupported"]},
        "reason": {"type": "string", "maxLength": 500},
        "steps": {"type": "array", "maxItems": 20, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["skill", "delta_mm", "seconds"],
            "properties": {
                "skill": {"type": "string", "enum": list(SKILLS)},
                "delta_mm": {"type": "array", "minItems": 3, "maxItems": 3,
                             "items": {"type": "number"}},
                "seconds": {"type": "number", "minimum": 0, "maximum": 10}
            }
        }}
    }
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_task(plan):
    """Strict allowlist. The model cannot set torque/speed/limits or execute code."""
    if not isinstance(plan, dict) or set(plan) != {"status", "reason", "steps"}:
        raise ValueError("计划必须恰好包含 status/reason/steps。")
    if plan["status"] not in ("plan", "unsupported"):
        raise ValueError("无效的计划状态。")
    if not isinstance(plan["reason"], str) or len(plan["reason"]) > 500:
        raise ValueError("reason 必须是 <=500 字符的字符串。")
    if not isinstance(plan["steps"], list) or len(plan["steps"]) > 20:
        raise ValueError("steps 必须是最多20步的列表。")
    if plan["status"] == "unsupported":
        if plan["steps"]:
            raise ValueError("unsupported 计划不能包含动作。")
        return plan
    if not plan["steps"]:
        raise ValueError("plan 需要至少一个技能。")
    for i, step in enumerate(plan["steps"], 1):
        if not isinstance(step, dict) or set(step) != {"skill", "delta_mm", "seconds"}:
            raise ValueError(f"第{i}步字段不符合技能协议。")
        skill, delta, seconds = step["skill"], step["delta_mm"], step["seconds"]
        if skill not in SKILLS or not isinstance(delta, list) or len(delta) != 3:
            raise ValueError(f"第{i}步技能或位移格式不正确。")
        if not all(finite(v) for v in delta) or not finite(seconds) or not 0 <= seconds <= 10:
            raise ValueError(f"第{i}步含无效数值。")
        if skill == "move_delta":
            if not 0 < motion.fk.norm(delta) <= 100 or seconds != 0:
                raise ValueError("move_delta 使用毫米向量，长度在 (0,100]，seconds=0。")
        elif any(delta):
            raise ValueError(f"{skill} 的 delta_mm 必须为 [0,0,0]。")
        if skill != "wait" and seconds != 0:
            raise ValueError(f"{skill} 的 seconds 必须为0。")
    # Round-trip produces a detached, plain data object; no injected callables.
    return json.loads(json.dumps(plan, ensure_ascii=False, allow_nan=False))


def demo_plan():
    return {"status": "plan", "reason": "空载组合实验：参考位、基座X正向60mm、等待、参考位；不是LLM生成。",
            "steps": [
                {"skill": "home", "delta_mm": [0,0,0], "seconds": 0},
                {"skill": "move_delta", "delta_mm": [60,0,0], "seconds": 0},
                {"skill": "wait", "delta_mm": [0,0,0], "seconds": 1},
                {"skill": "home", "delta_mm": [0,0,0], "seconds": 0}]}


def request_task(instruction, model="qwen3.5:9b", timeout=120.0):
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction)>2000:
        raise ValueError("指令不能为空，最多2000字符。")
    if not finite(timeout) or timeout <= 0:
        raise ValueError("timeout 必须为有限正数。")
    system = (
        "你是单臂空载运动规划器。仅输出符合JSON schema的数据，不输出代码或关节角。"
        "只允许move_delta、home、return_start、wait四个技能。"
        "move_delta的delta_mm=[dx,dy,dz]，单位毫米，坐标是base_link；1厘米=10毫米。"
        "每步位移长度最多100毫米。没有语言到相机或桌面坐标的标定。"
        "home表示用户已记录的参考姿态，不是电机全零。"
        "return_start表示本次任务开始前测量的姿态，不是最近一次home。"
        "wait的seconds是等待秒数（0到10）。其余技能seconds=0；"
        "除move_delta外delta_mm必须为[0,0,0]。"
        "最多20步。不可抓取、开合夹爪、自动换姿态逃避错误、修改限位/速度/供电。"
        "不知道当前场景坐标、物体位置或碰撞情况；不能声称看到物体。"
        "方向只支持用户明确给出的基座X/Y/Z正负方向；对含糊的前后左右上下不猜测。"
        "无法表示的目标返回status=unsupported、steps=[]并解释。"
        "不要添加用户没要求的home或额外动作；不要输出一套固定演示冒充理解。"
    )
    payload = {"model": model, "stream": False, "think": False, "format": SCHEMA,
               "messages": [{"role":"system","content":system},
                            {"role":"user","content":instruction}],
               "options":{"temperature":0},"keep_alive":"30m"}
    request = Request("http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type":"application/json"},method="POST")
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            reply = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.read().decode('utf-8',errors='replace')[:600]}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("本机 Ollama 请求失败；没有连接机械臂，也不会使用猜测的替代计划。") from exc
    if not isinstance(reply,dict) or reply.get("error") or reply.get("done") is not True:
        raise ValueError("Ollama 没有返回完整响应。")
    if reply.get("done_reason") in ("length","max_tokens","max_output_tokens"):
        raise ValueError("Ollama 输出被截断。")
    content = reply.get("message",{}).get("content")
    if not isinstance(content,str):
        raise ValueError("Ollama 响应中缺少 content。")
    return validate_task(json.loads(content))


def preflight_task(raw_start, calibration, task, home_raw=None,
                   ik_policy=None, execution_policy=None):
    """Compile every motion before the first write. No driver/hardware calls."""
    task = validate_task(task)
    if task["status"] != "plan":
        raise ValueError("不执行 unsupported 计划。")
    current = dict(raw_start)
    plans = []
    for i, step in enumerate(task["steps"],1):
        skill = step["skill"]
        if skill == "wait":
            plans.append({"index":i,"skill":skill,"seconds":step["seconds"]})
            continue
        if skill == "home" and home_raw is None:
            raise ValueError("任务需要参考位，但未提供本机已记录的参考文件。")
        target = home_raw if skill == "home" else raw_start if skill == "return_start" else None
        plan = motion.make_plan(current, step["delta_mm"], calibration,
                               ik_policy, execution_policy, joint_target_raw=target)
        plans.append({"index":i,"skill":skill,"plan":motion.plan_summary(plan)})
        current = dict(plan["path"][-1])
    return {"status":"TASK_PREFLIGHT_OK_MODEL_ONLY", "steps":plans,
            "initial_raw_position":dict(raw_start), "predicted_final_raw_position":current,
            "collision_checked":False, "physical_cartesian_accuracy_verified":False}


def observed_wait(bus, calibration, seconds, following_error_deg):
    """Wait while still reading the arm; don't sleep through a feedback loss."""
    anchor = motion.read_pose(bus)
    deadline = time.monotonic()+seconds
    while time.monotonic()<deadline:
        time.sleep(min(0.1,max(0.0,deadline-time.monotonic())))
        raw = motion.read_pose(bus)
        motion.check_pose(motion.to_degrees(raw,calibration), motion.recorded_bounds(calibration))
        if motion.max_error(raw,anchor)*motion.DEG_PER_TICK > following_error_deg:
            raise RuntimeError("等待期间机械臂发生明显位移；任务停止。")
    return {"status":"WAIT_COMPLETED", "seconds":seconds}


def execute_task(bus, calibration, task, *, home_raw=None, execute=False,
                 ik_policy=None, execution_policy=None, reporter=None):
    """Shared backend; serial ownership stays with the caller, including errors.

    register keys are capabilities, not arbitrary model-provided function names.
    Each motion replans from FRESH encoder readings; only confirmed results
    advance the task. Errors stop the remaining steps, never auto-return home.
    """
    task = validate_task(task)
    settings = (execution_policy or motion.ExecutionPolicy()).validate()
    raw_start,_ = motion.prepare_live(bus,calibration)
    preflight = preflight_task(raw_start,calibration,task,home_raw,ik_policy,settings)
    if not execute:
        return {"status":"TASK_PREVIEW_NO_WRITES", "task":task, "preflight":preflight}
    if reporter:
        reporter({"status":"TASK_READY", "task":task,"preflight":preflight})
    results, mutation = [], False
    def mark():
        nonlocal mutation
        mutation = True
    def do_move(step):
        return motion.move_ee(bus,calibration,delta_mm=step["delta_mm"],
            ik_policy=ik_policy,execution_policy=settings,execute=True,on_mutation=mark)
    def do_joint(target):
        return motion.move_ee(bus,calibration,joint_target_raw=target,
            ik_policy=ik_policy,execution_policy=settings,execute=True,on_mutation=mark)
    registry = {"move_delta":do_move, "home":lambda step:do_joint(home_raw),
                "return_start":lambda step:do_joint(raw_start),
                "wait":lambda step:observed_wait(bus,calibration,step["seconds"],settings.following_error_deg)}
    try:
        for i, step in enumerate(task["steps"],1):
            if reporter:
                reporter({"status":"SKILL_STARTED", "index":i, "skill":step["skill"]})
            result = registry[step["skill"]](step)
            record = {"index":i,"skill":step["skill"],"result":result}
            results.append(record)
            if reporter:
                reporter({"status":"SKILL_COMPLETED", **record})
    except (Exception,KeyboardInterrupt) as exc:
        details = {"completed_steps":results,"failed_step_index":len(results)+1,
                   "motion_was_attempted":mutation,"remaining_steps_executed":False}
        if mutation and bus.is_connected:
            details['hold_result'] = motion.attempt_hold(bus,calibration)
        if isinstance(exc,KeyboardInterrupt):
            exc.motion_report = details
            raise
        raise motion.MotionError("TASK_ABORTED",str(exc),**details) from exc
    return {"status":"TASK_COMPLETED_ENCODER_VERIFIED", "results":results,
            "motion_was_attempted":mutation,"object_task_success_verified":False,
            "physical_cartesian_accuracy_verified":False,
            "note":"仅确认空载运动与关节反馈；不是视觉验证、物体交换或新技能学习。"}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--demo',action='store_true')
    source.add_argument('--instruction')
    source.add_argument('--plan-file')
    parser.add_argument('--plan-only',action='store_true',help='只生成技能计划，不打开串口')
    parser.add_argument('--execute',action='store_true',help='实际执行；不能与 --plan-only 同用')
    parser.add_argument('--port',default='COM9')
    parser.add_argument('--id',default='friston_follower_12v')
    parser.add_argument('--calibration-dir')
    parser.add_argument('--home-file',default=str(Path(__file__).with_name('so101_zcd_home.json')))
    parser.add_argument('--model',default='qwen3.5:9b')
    parser.add_argument('--timeout',type=float,default=120.0)
    parser.add_argument('--speed-deg-s',type=float,default=8.0)
    args=parser.parse_args()
    bus=None
    try:
        motion.require(not(args.plan_only and args.execute),'--plan-only 与 --execute 不能同时使用。')
        settings=motion.ExecutionPolicy(speed_deg_s=args.speed_deg_s).validate()
        if args.instruction:
            task=request_task(args.instruction,args.model,args.timeout)
            source_name='LOCAL_OLLAMA'
        elif args.plan_file:
            task=validate_task(json.loads(Path(args.plan_file).read_text(encoding='utf-8-sig')))
            source_name='USER_JSON'
        else:
            task=validate_task(demo_plan())
            source_name='SCRIPTED_DEMO_NOT_LLM'
        motion.emit({'status':'PLAN_VALIDATED','source':source_name,'task':task})
        if task['status']=='unsupported':
            return 2
        if args.plan_only:
            return 0
        robot,calibration=motion.connect_robot(args)
        bus=robot.bus
        bus.connect()
        home_raw=(motion.home_reference(bus,calibration,args.id,args.home_file)
                  if any(s['skill']=='home' for s in task['steps']) else None)
        motion.emit(execute_task(bus,calibration,task,home_raw=home_raw,
            execute=args.execute,execution_policy=settings,reporter=motion.emit))
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        details=getattr(exc,'motion_report',getattr(exc,'details',{}))
        motion.emit({'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'TASK_STOPPED',
            'reason':str(exc) or 'Ctrl+C','code':getattr(exc,'code','INPUT_OR_CONNECTION_ERROR'),
            'motion_was_attempted':False,**details})
        return 130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        if bus is not None and bus.is_connected:
            try:
                bus.disconnect(disable_torque=False)
            except Exception as exc:
                print(f'串口关闭失败；请检查设备状态：{exc}')


if __name__=='__main__':
    raise SystemExit(main())
