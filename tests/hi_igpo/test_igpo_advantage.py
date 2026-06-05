import torch
from verl.trainer.ppo.core_algos import compute_igpo_turn_advantage

# ───────────────────────────────────────────────────────────────────────────
# 方案 A:IGPO 原生 token 级签名
#   compute_igpo_turn_advantage(token_level_rewards, response_mask, index,
#                               turn_boundary_mask, gamma, info_gain_norm_mode,
#                               ig_group_mode, min_group_size, norm_by_std)
# 约定(= IGPO):IG 写在每个 turn 末 token,F1 写在每行最后一个有效 token;
#   turn_boundary_mask 标全部 turn 末位置(含 F1);函数内部 f1_mask=每行最后有效
#   token,ig_mask = turn_boundary_mask & ~f1_mask。turn_group 的 turn-index 由
#   每行 ig_mask 的出现顺序推出(第 k 个 IG 边界 = turn k)。
# 期望值与重构前的 turn_records 版完全相同(IGPO 公式真值不变)。
# ───────────────────────────────────────────────────────────────────────────

# 两条同 prompt 轨迹,各 2 个 IG 中间轮 + 1 个 F1 末轮,response_len=6,每轮占 2 token:
#   IG 写在 pos1/pos3,F1 写在 pos5
#   token_level_rewards 行0=[0,0.3,0,0.1,0,1.0] 行1=[0,0.1,0,0.5,0,0.0]
def _two_traj_6tok():
    tlr = torch.tensor([[0., 0.3, 0., 0.1, 0., 1.0],
                        [0., 0.1, 0., 0.5, 0., 0.0]])
    mask = torch.ones(2, 6)
    boundary = torch.tensor([[False, True, False, True, False, True],
                             [False, True, False, True, False, True]])
    index = torch.tensor([0, 0])
    return tlr, mask, boundary, index


# 期望值按 IGPO 公式(separate 归一化 + mean-only + gamma=1)手算:
#   IG 池 [0.3,0.1,0.1,0.5] mean=0.25 → norm IG=[0.05,-0.15,-0.15,0.25]
#   F1 池 [1.0,0.0] mean=0.5         → norm F1=[0.5,-0.5]
#   归一化后 turn 优势 A=[0.05,-0.15,0.5], B=[-0.15,0.25,-0.5]
#   gamma=1 从后往前累加:A=[0.4,0.35,0.5], B=[-0.4,-0.25,-0.5]
#   散到 token span [0:2,2:4,4:6]
def test_port_global_separate_matches_igpo_formula():
    tlr, mask, boundary, index = _two_traj_6tok()
    adv, ret = compute_igpo_turn_advantage(
        tlr, response_mask=mask, index=index, turn_boundary_mask=boundary,
        ig_group_mode="global", info_gain_norm_mode="separate",
        norm_by_std=False, gamma=1.0)
    assert torch.allclose(adv[0], torch.tensor([0.4, 0.4, 0.35, 0.35, 0.5, 0.5]), atol=1e-6)
    assert torch.allclose(adv[1], torch.tensor([-0.4, -0.4, -0.25, -0.25, -0.5, -0.5]), atol=1e-6)
    assert torch.allclose(ret, adv)


# norm_by_std=True characterization test — pins the production default path.
# Expected values hand-computed from IGPO formula (population variance, sqrt(var+1e-8)):
#   IG pool [0.3,0.1,0.1,0.5]: mean=0.25, var=0.0275, std≈0.165831
#   F1 pool [1.0,0.0]: mean=0.5, var=0.25, std≈0.5
#   gamma=1 backward accumulation per traj, then broadcast to each turn's span.
def test_port_global_separate_norm_by_std():
    tlr, mask, boundary, index = _two_traj_6tok()
    adv, ret = compute_igpo_turn_advantage(
        tlr, response_mask=mask, index=index, turn_boundary_mask=boundary,
        ig_group_mode="global", info_gain_norm_mode="separate",
        norm_by_std=True, gamma=1.0)
    assert torch.allclose(adv[0], torch.tensor([0.396977, 0.396977, 0.095466, 0.095466, 1.0, 1.0]), atol=1e-4)
    assert torch.allclose(adv[1], torch.tensor([-0.396977, -0.396977, 0.507557, 0.507557, -1.0, -1.0]), atol=1e-4)
    assert torch.allclose(ret, adv)


def test_turn_group_separate_matches_handcalc():
    # 按 (prompt, turn-index) 分组,turn-index 由每行 IG 边界顺序推出:pos1=t0, pos3=t1
    #   IG t0 组 [0.3,0.1] mean=0.2 ; IG t1 组 [0.1,0.5] mean=0.3 ; F1 组 mean=0.5
    #   归一化后 turn 优势 A=[0.1,-0.2,0.5], B=[-0.1,0.2,-0.5]
    #   gamma=1 累加:A=[0.4,0.3,0.5], B=[-0.4,-0.3,-0.5]
    tlr, mask, boundary, index = _two_traj_6tok()
    adv, _ = compute_igpo_turn_advantage(
        tlr, response_mask=mask, index=index, turn_boundary_mask=boundary,
        ig_group_mode="turn_group", info_gain_norm_mode="separate",
        norm_by_std=False, gamma=1.0)
    assert torch.allclose(adv[0], torch.tensor([0.4, 0.4, 0.3, 0.3, 0.5, 0.5]), atol=1e-6)
    assert torch.allclose(adv[1], torch.tensor([-0.4, -0.4, -0.3, -0.3, -0.5, -0.5]), atol=1e-6)


def test_turn_group_singleton_falls_back_to_prompt():
    # 变长:Traj A 有 3 个 IG turn,Traj B 只有 2 个 → (prompt, t2) 组单样本 → 回退到 IG prompt 级统计。
    # response_len=8:A 行 IG@pos1/3/5 + F1@pos7;B 行 IG@pos1/3 + F1@pos5(pos6,7 padding,mask=0)。
    #   IG 池 [0.3,0.1,0.9,0.1,0.5] mean=0.38 ; F1 池 [1.0,0.0] mean=0.5
    #   norm(norm_by_std=False):
    #     (p,t0)[0.3,0.1]mean0.2 → A=0.1, B=-0.1 ; (p,t1)[0.1,0.5]mean0.3 → A=-0.2, B=0.2
    #     (p,t2) 单样本→回退 IG prompt 均值0.38 → A t2 = 0.9-0.38 = 0.52
    #     F1 mean0.5 → A=+0.5, B=-0.5
    #   累加 gamma=1:
    #     A: F1=0.5; t2=0.52+0.5=1.02; t1=-0.2+1.02=0.82; t0=0.1+0.82=0.92
    #        散到 span [0:2,2:4,4:6,6:8] → [0.92,0.92,0.82,0.82,1.02,1.02,0.5,0.5]
    #     B: F1=-0.5; t1=0.2-0.5=-0.3; t0=-0.1-0.3=-0.4
    #        散到 span [0:2,2:4,4:6] → [-0.4,-0.4,-0.3,-0.3,-0.5,-0.5,0,0](pos6,7 masked=0)
    # 关键:若回退失效,单样本 mean=自身0.9 → norm=0 → t2_adv=0+0.5=0.5≠1.02,故 1.02 钉死了回退生效。
    tlr = torch.tensor([[0., 0.3, 0., 0.1, 0., 0.9, 0., 1.0],
                        [0., 0.1, 0., 0.5, 0., 0.0, 0., 0.0]])
    mask = torch.tensor([[1., 1, 1, 1, 1, 1, 1, 1],
                         [1., 1, 1, 1, 1, 1, 0, 0]])
    boundary = torch.tensor([[False, True, False, True, False, True, False, True],
                             [False, True, False, True, False, True, False, False]])
    index = torch.tensor([0, 0])
    adv, _ = compute_igpo_turn_advantage(
        tlr, response_mask=mask, index=index, turn_boundary_mask=boundary,
        ig_group_mode="turn_group", info_gain_norm_mode="separate",
        norm_by_std=False, gamma=1.0, min_group_size=2)
    assert not torch.isnan(adv).any()
    assert torch.allclose(adv[0], torch.tensor([0.92, 0.92, 0.82, 0.82, 1.02, 1.02, 0.5, 0.5]), atol=1e-6)
    assert torch.allclose(adv[1], torch.tensor([-0.4, -0.4, -0.3, -0.3, -0.5, -0.5, 0., 0.]), atol=1e-6)


def test_turn_group_separate_norm_by_std():
    # turn_group + norm_by_std=True 生产默认路径(总体方差 + 1e-8):
    #   IG t0 [0.3,0.1] mean0.2 std≈0.1 → A=+1.0, B=-1.0
    #   IG t1 [0.1,0.5] mean0.3 std≈0.2 → A=-1.0, B=+1.0
    #   F1     [1.0,0.0] mean0.5 std≈0.5 → A=+1.0, B=-1.0
    #   累加: A backward F1=1.0,t1=-1.0+1.0=0.0,t0=1.0+0.0=1.0 → [1,1,0,0,1,1]
    #         B backward F1=-1.0,t1=1.0-1.0=0.0,t0=-1.0+0.0=-1.0 → [-1,-1,0,0,-1,-1]
    tlr, mask, boundary, index = _two_traj_6tok()
    adv, _ = compute_igpo_turn_advantage(
        tlr, response_mask=mask, index=index, turn_boundary_mask=boundary,
        ig_group_mode="turn_group", info_gain_norm_mode="separate",
        norm_by_std=True, gamma=1.0)
    assert torch.allclose(adv[0], torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0]), atol=1e-4)
    assert torch.allclose(adv[1], torch.tensor([-1.0, -1.0, 0.0, 0.0, -1.0, -1.0]), atol=1e-4)
