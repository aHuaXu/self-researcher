import torch
from verl.trainer.ppo.core_algos import compute_igpo_turn_advantage

# 两条同 prompt 轨迹,各 2 个 IG 中间轮 + 1 个 F1 末轮
#   Traj A: IG=[0.3,0.1], F1=1.0 ; Traj B: IG=[0.1,0.5], F1=0.0
# turn_records 字段全为 torch.Tensor,构造按 Task 0 笔记里 IGPO 的入参表示对齐;
# 期望值按 IGPO 公式(separate 归一化 + mean-only + gamma=1)手算:
#   IG 池 [0.3,0.1,0.1,0.5] mean=0.25 → norm IG=[0.05,-0.15,-0.15,0.25]
#   F1 池 [1.0,0.0] mean=0.5         → norm F1=[0.5,-0.5]
#   归一化后 A=[0.05,-0.15,0.5], B=[-0.15,0.25,-0.5]
#   gamma=1 从后往前累加:A=[0.4,0.35,0.5], B=[-0.4,-0.25,-0.5]
#   散到 token span [0:2,2:4,4:6]
def test_port_global_separate_matches_igpo_formula():
    rec = dict(
        turn_reward=torch.tensor([0.3,0.1,1.0,0.1,0.5,0.0]),
        prompt_id  =torch.tensor([0,0,0,0,0,0]),
        traj_id    =torch.tensor([0,0,0,1,1,1]),
        turn_pos   =torch.tensor([0,1,2,0,1,2]),
        is_outcome =torch.tensor([False,False,True,False,False,True]),
        sample_row =torch.tensor([0,0,0,1,1,1]),
        span_start =torch.tensor([0,2,4,0,2,4]),
        span_end   =torch.tensor([2,4,6,2,4,6]),
    )
    adv, ret = compute_igpo_turn_advantage(
        rec, bs=2, response_len=6,
        ig_group_mode="global", info_gain_norm_mode="separate",
        norm_by_std=False, gamma=1.0)
    assert torch.allclose(adv[0], torch.tensor([0.4,0.4,0.35,0.35,0.5,0.5]), atol=1e-6)
    assert torch.allclose(adv[1], torch.tensor([-0.4,-0.4,-0.25,-0.25,-0.5,-0.5]), atol=1e-6)
    assert torch.allclose(ret, adv)


# norm_by_std=True characterization test — pins the production default path.
# Expected values hand-computed from IGPO formula (population variance, sqrt(var+1e-8)):
#   IG pool [0.3,0.1,0.1,0.5]: mean=0.25, var=0.0275, std≈0.165831
#     norm: 0.301509, -0.904528, -0.904528, 1.507547   (divided by std+1e-6)
#   F1 pool [1.0,0.0]: mean=0.5, var=0.25, std≈0.5
#     norm: ~1.0, ~-1.0                                  (divided by std+1e-6≈0.500001)
#   gamma=1 backward accumulation per traj, then broadcast to each turn's span:
#     Traj A: turn0_adv≈0.396981, turn1_adv≈0.095472, turn2_adv≈1.0
#     Traj B: turn0_adv≈-0.396981, turn1_adv≈0.507547, turn2_adv≈-1.0
def test_port_global_separate_norm_by_std():
    rec = dict(
        turn_reward=torch.tensor([0.3,0.1,1.0,0.1,0.5,0.0]),
        prompt_id  =torch.tensor([0,0,0,0,0,0]),
        traj_id    =torch.tensor([0,0,0,1,1,1]),
        turn_pos   =torch.tensor([0,1,2,0,1,2]),
        is_outcome =torch.tensor([False,False,True,False,False,True]),
        sample_row =torch.tensor([0,0,0,1,1,1]),
        span_start =torch.tensor([0,2,4,0,2,4]),
        span_end   =torch.tensor([2,4,6,2,4,6]),
    )
    adv, ret = compute_igpo_turn_advantage(
        rec, bs=2, response_len=6,
        ig_group_mode="global", info_gain_norm_mode="separate",
        norm_by_std=True, gamma=1.0)
    assert torch.allclose(adv[0], torch.tensor([0.396977,0.396977,0.095466,0.095466,1.0,1.0]), atol=1e-4)
    assert torch.allclose(adv[1], torch.tensor([-0.396977,-0.396977,0.507557,0.507557,-1.0,-1.0]), atol=1e-4)
    assert torch.allclose(ret, adv)


def test_turn_group_separate_matches_handcalc():
    # 同 Task1 的两条轨迹,但按 (prompt,turn_pos) 分组归一化:
    #   IG t0 组 [0.3,0.1] mean=0.2 ; IG t1 组 [0.1,0.5] mean=0.3 ; F1 组 mean=0.5
    #   归一化后 A=[0.1,-0.2,0.5], B=[-0.1,0.2,-0.5]
    #   gamma=1 累加:A=[0.4,0.3,0.5], B=[-0.4,-0.3,-0.5]
    rec = dict(
        turn_reward=torch.tensor([0.3,0.1,1.0,0.1,0.5,0.0]),
        prompt_id  =torch.tensor([0,0,0,0,0,0]),
        traj_id    =torch.tensor([0,0,0,1,1,1]),
        turn_pos   =torch.tensor([0,1,2,0,1,2]),
        is_outcome =torch.tensor([False,False,True,False,False,True]),
        sample_row =torch.tensor([0,0,0,1,1,1]),
        span_start =torch.tensor([0,2,4,0,2,4]),
        span_end   =torch.tensor([2,4,6,2,4,6]),
    )
    adv, _ = compute_igpo_turn_advantage(
        rec, bs=2, response_len=6,
        ig_group_mode="turn_group", info_gain_norm_mode="separate",
        norm_by_std=False, gamma=1.0)
    assert torch.allclose(adv[0], torch.tensor([0.4,0.4,0.3,0.3,0.5,0.5]), atol=1e-6)
    assert torch.allclose(adv[1], torch.tensor([-0.4,-0.4,-0.3,-0.3,-0.5,-0.5]), atol=1e-6)

def test_turn_group_singleton_falls_back_to_global():
    # Traj A 多一个 t2 的 IG 轮(B 没有)→ (prompt,t2) 组只有 1 个 → 回退全局 IG 统计
    # 全 IG(无 F1 末轮),全局 IG mean=(0.3+0.1+0.9+0.1+0.5)/5=0.38
    #   t2 组单样本 → 用全局 → 0.9-0.38=0.52(末轮无累加后继,Ã2=0.52)
    rec = dict(
        turn_reward=torch.tensor([0.3,0.1,0.9,0.1,0.5]),
        prompt_id  =torch.tensor([0,0,0,0,0]),
        traj_id    =torch.tensor([0,0,0,1,1]),
        turn_pos   =torch.tensor([0,1,2,0,1]),
        is_outcome =torch.tensor([False,False,False,False,False]),
        sample_row =torch.tensor([0,0,0,1,1]),
        span_start =torch.tensor([0,2,4,0,2]),
        span_end   =torch.tensor([2,4,6,2,4]),
    )
    adv, _ = compute_igpo_turn_advantage(
        rec, bs=2, response_len=6,
        ig_group_mode="turn_group", info_gain_norm_mode="separate",
        norm_by_std=False, gamma=1.0, min_group_size=2)
    # 行0 t2 token span [4:6] 的优势 = 0.52
    assert torch.allclose(adv[0, 4:6], torch.tensor([0.52, 0.52]), atol=1e-6)


def test_turn_group_separate_norm_by_std():
    # 钉住 turn_group + norm_by_std=True 的生产默认路径。
    # 使用与 test_turn_group_separate_matches_handcalc 相同的 6-turn 输入。
    # 期望值(统一后公式手算,总体方差 + 1e-8):
    #   IG t0 组 [0.3,0.1]: mean=0.2, var=0.01, std=sqrt(0.01+1e-8)≈0.1  → A=+1.0, B=-1.0
    #   IG t1 组 [0.1,0.5]: mean=0.3, var=0.04, std=sqrt(0.04+1e-8)≈0.2  → A=-1.0, B=+1.0
    #   F1  组  [1.0,0.0]: mean=0.5, var=0.25, std=sqrt(0.25+1e-8)≈0.5  → A=+1.0, B=-1.0
    #   gamma=1 累加: A=[1.0+0.0+1.0, 0.0+1.0, 1.0] = [2.0, 1.0, 1.0]
    #     → 从后往前:turn2=1.0, turn1=0.0+1.0=1.0 ? 不对,重新理清:
    #     A 三个 turn 的归一化: IG_t0=+1.0, IG_t1=-1.0, F1=+1.0
    #     backward: turn2_adv=1.0, turn1_adv=-1.0+1.0*1.0=0.0, turn0_adv=1.0+1.0*0.0=1.0
    #     散到 span: [1.0,1.0, 0.0,0.0, 1.0,1.0]
    #   B: IG_t0=-1.0, IG_t1=+1.0, F1=-1.0
    #     turn2=-1.0, turn1=1.0+(-1.0)=0.0, turn0=-1.0+0.0=-1.0
    #     散到 span: [-1.0,-1.0, 0.0,0.0, -1.0,-1.0]
    rec = dict(
        turn_reward=torch.tensor([0.3, 0.1, 1.0, 0.1, 0.5, 0.0]),
        prompt_id  =torch.tensor([0, 0, 0, 0, 0, 0]),
        traj_id    =torch.tensor([0, 0, 0, 1, 1, 1]),
        turn_pos   =torch.tensor([0, 1, 2, 0, 1, 2]),
        is_outcome =torch.tensor([False, False, True, False, False, True]),
        sample_row =torch.tensor([0, 0, 0, 1, 1, 1]),
        span_start =torch.tensor([0, 2, 4, 0, 2, 4]),
        span_end   =torch.tensor([2, 4, 6, 2, 4, 6]),
    )
    adv, _ = compute_igpo_turn_advantage(
        rec, bs=2, response_len=6,
        ig_group_mode="turn_group", info_gain_norm_mode="separate",
        norm_by_std=True, gamma=1.0)
    assert torch.allclose(adv[0], torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0]), atol=1e-4)
    assert torch.allclose(adv[1], torch.tensor([-1.0, -1.0, 0.0, 0.0, -1.0, -1.0]), atol=1e-4)
