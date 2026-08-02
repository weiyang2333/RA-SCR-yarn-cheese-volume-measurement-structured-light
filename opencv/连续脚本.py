import subprocess
import sys
import os

def run_step(script_path, step_name, extra_args=None, work_dir=None):
    print(f"\n====== 开始执行: {step_name} ======")
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=work_dir)
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} 执行失败，返回码: {result.returncode}")
    print(f"====== 完成: {step_name} ======\n")



##本脚本是针对圆台转动时快速拍摄保存，调试请不要运行
if __name__ == "__main__":
    Haikang_callback_automation = r"..\opencv\Haikang_callback_automation.py"
    phase_shift_acquire = r"..\en_decode\decode_multi_phase.py"
    cloud_acquire = r"..\en_decode\Triangulated_point_cloud.py"

    run_step(Haikang_callback_automation,"拍摄",work_dir=os.path.dirname(Haikang_callback_automation))
    run_step(phase_shift_acquire,"相位提取",work_dir=os.path.dirname(phase_shift_acquire))
    cloud_idx = input("请输入本次点云编号: ").strip()
    run_step(cloud_acquire,"点云生成",[cloud_idx],work_dir=os.path.dirname(cloud_acquire))

    print("全部流程执行完毕,继续转动圆台")