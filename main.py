"""程序入口：创建共享设备，再按输入、身份、技能和回复呈现运行接待循环。

控制台语法在 friston3.console；模型规划与技能分发仍交给 Agent。
"""
import argparse

from friston3.npc.identity import IDENTITY
from friston3.console import (
    discard_input, read_instruction, new_visitor, handle_identity_command,
    run_device_command, handle_session_command, print_help,
)


def _arguments():
    parser = argparse.ArgumentParser(description="Friston-3 漫展终端")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--download-owner-models", choices=("face", "voice", "all"),
                           help="显式联网下载身份模型；不启动硬件或对话")
    operation.add_argument("--owner-status", action="store_true",
                           help="查看本地身份模板和模型状态；不启动硬件")
    operation.add_argument("--forget-owner", action="store_true",
                           help="删除主人人脸与声纹模板")
    operation.add_argument("--enroll-owner", choices=("face", "voice", "both"),
                           help="在本地控制台录入主人；不加载 ASR、对话或 TTS")
    return parser.parse_args()


def main():
    arguments = _arguments()
    face = camera = speech_input = owner = None
    try:
        if not any(vars(arguments).values()):
            from friston3.runtime_bootstrap import use_installed_runtime
            child_status = use_installed_runtime()
            if child_status is not None:
                return child_status
        # Setup/status paths return before creating device workers or chat models.
        if arguments.download_owner_models:
            from friston3.owner_setup import download_owner_models
            prepared = download_owner_models(arguments.download_owner_models)
            return 0 if prepared["success"] else 1

        from friston3.owner_identity import OwnerIdentityManager, OwnerProfile

        if arguments.forget_owner:
            OwnerProfile().forget()
            print("主人人脸与声纹模板已删除；对话记忆保持原状。")
            return 0
        if arguments.owner_status:
            owner = OwnerIdentityManager()
            print(owner.status_text())
            return 0

        from friston3.hardware.camera import create_camera
        from stackchan_integration import StackChanFaceEvents

        # Camera, microphone, expressions and playback share one USB/TCP owner.
        face = StackChanFaceEvents()
        camera = create_camera(face)
        owner = OwnerIdentityManager(camera=camera, device=face)
        if arguments.enroll_owner:
            return 0 if owner.enroll(arguments.enroll_owner) else 1

        print(f"{IDENTITY.name} 启动（栈酱摄像头 / 栈酱麦克风 / 自动连接）")
        print("[StackChan] 自动发现 USB 或本地 Wi-Fi；设备尚未连接时会继续等待。")
        owner.warmup()
        from stackchan_integration.voice_input import StackChanSpeechInput
        speech_input = StackChanSpeechInput(face)
        speech_input.audio_observer = owner.begin_utterance
        speech_input.preload()
        _run_session(face, camera, speech_input, owner)
        return 0
    except (EOFError, KeyboardInterrupt):
        print(f"\n{IDENTITY.name} 已退出。")
        return 0
    except Exception as error:
        # Include dependency/configuration failures in the visible startup
        # boundary. No device operation or model download is retried silently.
        print(f"[{IDENTITY.name}] 启动或运行未完成：{type(error).__name__}: {error}")
        print("请检查上面指明的本地模型或依赖；普通设备断线会自动恢复。")
        return 1
    finally:
        for resource in (owner, speech_input, camera, face):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    print(f"[{IDENTITY.name}] 资源释放未确认：{type(error).__name__}: {error}")


def _run_session(face, camera, speech_input, owner):
    """装配技能并管理一轮互动的顺序；退出时关闭语音和机械臂联动。"""
    from friston3.agent import Agent
    from friston3.skills.auto_photo import AutoPhotoSkill
    from friston3.skills.photo import PhotoSkill
    from friston3.skills.manager import SkillManager
    from friston3.skills.npc import NpcTalkSkill
    from friston3.skills.volume import VolumeControlSkill
    from friston3.skills.device_settings import DeviceSettingsSkill
    from friston3.skills.arms import ArmsControlSkill
    from friston3.arms.integration import ArmIntegration, is_immediate_stop
    from friston3.vision.yunet_observer import YuNetObserver
    from friston3.planner import LLMPlanner
    from friston3.input_text import prepare_input_text, is_memory_command
    from stackchan_integration.voice_output import AgentSpeechOutput

    # Warm the resident TTS engine while the dialogue models initialize, and
    # give the volume skill the very same output object used for all replies.
    voice = AgentSpeechOutput(face)
    arms = ArmIntegration()
    try:
        manager = SkillManager()
        manager.register(PhotoSkill(camera))
        manager.register(AutoPhotoSkill(camera, observer=YuNetObserver()))
        manager.register(VolumeControlSkill(voice))
        manager.register(DeviceSettingsSkill(face))
        manager.register(ArmsControlSkill(arms))
        npc = NpcTalkSkill()
        manager.register(npc)
        agent = Agent(manager, LLMPlanner(manager))

        print_help(speech_enabled=speech_input is not None)

        while True:
            try:
                raw_instruction, input_source = read_instruction(owner, speech_input)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{IDENTITY.name} 已退出。")
                break
            if not raw_instruction:
                discard_input(owner, speech_input)
                continue
            # /t is an optional local spelling for one typed utterance.
            if input_source == "text" and raw_instruction.lower().startswith("/t "):
                raw_instruction = raw_instruction[3:].strip()
            instruction = prepare_input_text(
                raw_instruction, input_source=input_source)["normalized_text"]
            command = instruction.rstrip("。！？!?.,，；; ").lower()
            if not instruction:
                discard_input(owner, speech_input)
                continue

            # Stop has no identity/LLM precondition once input has arrived.
            # The microphone remains half-duplex: use the independent service
            # CLI while this thread is occupied by planning or playback.
            if is_immediate_stop(instruction):
                discard_input(owner, speech_input)
                stopped = manager.execute("control_arms", {"operation": "stop"})
                print(f"[Arms] {stopped['message']}")
                continue

            if input_source == "text":
                if command.split()[:1] in (["/arms"], ["/voice"], ["/device"]):
                    discard_input(owner, speech_input)
                    run_device_command(command, manager, face)
                    continue
                # The physical console can inspect the actual previous result
                # without replacing a visitor's trace with a new owner turn.
                if command in ("查看证据", "/evidence"):
                    discard_input(owner, speech_input)
                    print(npc.explain_last_response())
                    continue
                if handle_identity_command(command, owner, speech_input, npc):
                    continue
            elif command.split()[:1] in (["/owner"], ["/visitor"]):
                discard_input(owner, speech_input)
                print(f"{IDENTITY.name}: 这项管理操作需要在电脑控制台输入。")
                continue

            if command in ("下一位来访者", "下一位访客", "新访客"):
                new_visitor(owner, speech_input, npc)
                continue

            pcm = speech_input.last_audio_pcm if input_source == "speech" else None
            sample_rate = speech_input.last_audio_rate if input_source == "speech" else 16000
            try:
                context = owner.context_for(input_source=input_source, pcm=pcm, sample_rate=sample_rate)
            finally:
                if speech_input is not None:
                    speech_input.clear_last_audio()
            npc.set_interlocutor(context)
            label = "主人 / 可露希尔" if context["role"] == "owner" else (
                "来访者" if context["role"] == "visitor" else "身份未确认")
            print(f"[身份] {label}；依据={context['source']}；额外等待={context['wait_ms']:.0f}ms")

            # Spoken management sees this utterance's context, never the prior
            # speaker's private history or authorization.
            action = handle_session_command(command, npc, input_source=input_source, context=context)
            if action == "exit":
                break
            if action == "handled":
                continue

            face.on_request()
            try:
                result = agent.handle(raw_instruction, input_source=input_source, interlocutor=context)
                executed = result.get("executed_skill")
                if (executed == "configure_device" and result.get("success") is True
                        and result.get("setting") == "gestures" and result.get("choice") == "off"):
                    arm_stop = arms.cancel_dialogue()
                    if arm_stop is not None:
                        result["arm_motion_stop"] = arm_stop
                        if arm_stop.get("success") is not True or arm_stop.get("status") == "stopping":
                            result["message"] += " 机械臂：" + arm_stop["message"]
                if executed in ("take_photo", "auto_photo"):
                    npc.record_execution(executed, result)
                reply_text = str(result.get("message", "任务结束"))
                if (executed in ("take_photo", "auto_photo")
                        and result.get("success") is True and context["role"] == "owner"):
                    # Presentation only: retain original device result, saved
                    # path, success flag and execution record without changes.
                    reply_text = "可露希尔，" + reply_text
                print(f"{IDENTITY.name}:", reply_text)
                face.prepare_reply(result)
                arm_start = arms.reply_start(result, allow=not is_memory_command(instruction))
                if ((result.get("success", True) or executed == "control_arms")
                        and ((executed == "npc_talk" and not is_memory_command(instruction))
                             or (executed in ("control_volume", "configure_device", "control_arms")
                                 and result.get("speak_confirmation")))):
                    voice.speak(reply_text, on_start=arm_start)
                if arm_start is not None:
                    arm_start()
                # Speech consumes the gesture just before the first clip.
                # A muted/unavailable voice still gets one bounded gesture.
                face.on_task_completed(result)
            finally:
                face.on_reply_ready()
    finally:
        arms.close()
        voice.close()
        discard_input(owner, speech_input)


if __name__ == "__main__":
    raise SystemExit(main())
