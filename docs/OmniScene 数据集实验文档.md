# OmniScene 数据集实验文档

本文记录已审阅通过的实验协议及对应实现。2026-09-25 已实现配置、数据适配、静态训练与评估，并完成 CPU 回归测试和受限 GPU 冒烟检查；没有启动正式的 100,001 步训练或完整测试。使用方法与已验证范围见第 10 节。

代码核对基线（2026-09-24）：GaussianSTORM 原作者代码 `fcd2561`，本地环境配置提交 `2b4e87a`；SVF-GS `main@af39b31`；depthsplat `comp_svfgs@405b9a5`。环境配置已按要求以“4090环境配置”提交并推送至 `origin/main`，本文在随后创建的 `comp_svfgs` 分支撰写。

## 1. 实验目标及已确认的选择

在 OmniScene 上从六路环视 RGB 和相机参数一次前向预测场景高斯，分别渲染 18 路目标与其中 12 路新视角，与 SVF-GS、depthsplat 比较。每个 bin 独立重建，不累积跨 bin 状态，不输入历史帧，不使用真实时间、场景流或运动标签。

| 项目 | 已确认的方案 |
| --- | --- |
| 模型 | 标准像素版 **STORM-B/8**，`gs_dim=3`，`decoder_type=dummy` |
| 分辨率 | **112×200、224×400**，各自独立训练、保存及评估 |
| 初始化 | 两个实验均从头训练，不加载 STORM 或 Latent-STORM 预训练权重 |
| 时间和运动模块 | 保留原模块及 16 个 motion tokens；全部时间固定为零，并显式禁止高斯位移 |
| 天空和颜色模块 | 保留由输入 RGB 学习的 sky token；使用 6 个相机 affine tokens |
| 深度监督 | 18 路目标的已有 Metric3D 尺度深度；仅用于监督，不加入编码器输入 |
| 天空监督 | 不读取天空掩码；关闭依赖天空掩码的所有损失 |
| PCC 参考 | 已有 DA2 相对深度，只在验证/测试指标侧读取 |
| PCC 渲染深度 | 对齐 depthsplat：累积的相机 z 深度；训练继续使用 STORM 期望 z 深度 |
| 学习率 | 原生 cosine，峰值/配置值 `0.0004`，warmup 5,000 步 |
| Batch size | 训练、验证、测试均为 1；本方案单 GPU、无梯度累积，全局 batch size 也是 1 |
| 训练长度 | 完成 **100,001 次 optimizer 更新** |
| 评估节奏 | 每 1,000 步验证；每 10 次验证进行一次 mini 测试；训练结束必须再做 mini 测试 |
| 日志 | W&B offline，同时保留本地配置、逐 bin 指标和汇总文件；训练启动和 mini 测试完成后推送飞书 |

本项目没有可直接沿用的 RE10K 实验配置；RE10K 是 depthsplat 的配置来源，不能据此替换 STORM 的编码器、损失或学习率。用户已选择标准 STORM-B/8，因此不再比较 Latent-STORM、Large 模型或不同预训练初始化。

动态掩码覆盖范围已确认：沿用两个参考加载器，仅对 12 路新视角读取真实掩码，末尾 6 路输入视角目标的掩码全为有效；`use_dynamic_mask=true`，18 路深度监督均保留。

### 1.1 数据使用边界

| 数据 | 重建网络输入 | 训练监督 | 验证/测试 |
| --- | --- | --- | --- |
| 中心六路 RGB、相机内外参 | 是 | 其中 RGB 也是最后六路目标真值 | 重建输入及相应目标真值 |
| 十二路新视角 RGB、相机参数 | RGB 不输入编码器；相机参数仅供目标渲染 | RGB 重建监督 | 目标渲染及图像指标 |
| 18 路 Metric3D 深度 | 否 | 是，按有效像素计算深度损失 | 可以报告辅助深度损失；不充当 PCC 参考 |
| 已有动态物体掩码 | 不增加输入通道 | 是，约束 RGB/LPIPS/深度损失的有效区域 | 主评估使用完整图像，不用训练掩码过滤指标 |
| 18 路已有 DA2 深度 | 否 | 否，训练 loader 不读取 | 仅 PCC |
| 时间戳、相邻帧输入、LiDAR 点云/深度、天空掩码、光流、Metric3D 置信度 | 不使用 | 不使用 | 不使用 |

不得运行预处理程序生成或补齐额外资产。已有掩码即使包含历史生产流程的信息，本实验也只读取允许的最终掩码，不访问其生产依据。网络内部预测的 rays、深度、高斯、motion tokens 和 sky 特征不属于额外离线信息。

十二路新视角目标虽来自其他采集位置/时刻，但这里只使用已给定的相机位姿和监督图像；不向模型提供这些目标图像，也不传入真实时间或时间差。目标选择索引 1/2 不作为时间编码。

## 2. 配置组织与加载

### 2.1 保留本项目入口，增加轻量配置层

保留 `main_storm.py` 的 argparse、模型工厂、AdamW 和训练循环；增加 YAML 配置读取，不引入 depthsplat 的 Hydra/Lightning，也不依赖运行时导入另外两个项目。

配置文件：

```text
configs/dataset/omniscene.yaml
configs/experiment/omniscene_base.yaml
configs/experiment/omniscene_112x200.yaml
configs/experiment/omniscene_224x400.yaml
```

加载顺序：原生参数默认值 → 数据集配置 → 公共实验配置 → 分辨率实验配置 → 命令行显式覆盖。`extends` 和 `dataset_config` 相对引用它的配置文件解析；数据及输出路径统一相对项目根目录解析，也支持绝对路径。用 `yaml.safe_load`，检查未知字段、循环继承及冲突；命令行未显式指定的默认值不能覆盖 YAML。

启动后保存最终展开的 `resolved_config.yaml`、实际命令、代码 SHA、随机种子、数据清单哈希及选中 bin 列表。两种分辨率独立输出目录，恢复训练时检查模型、分辨率和数据协议一致。

以下配置字段已接入 `storm/omniscene_config.py`，由 `main_storm.py` 分派到 `storm/omniscene_runner.py`，分别传入模型构造、loader、训练循环和评估器。原作者的数据集仍使用原训练入口逻辑。

### 2.2 数据集配置

```yaml
# configs/dataset/omniscene.yaml
dataset: omniscene
data_root: /home/B_UserData/dongzhipeng/Datasets/dataset_omniscene
data_version: interp_12Hz_trainval
train_manifest: bins_train_3.2m.json
eval_manifest: bins_val_3.2m.json
camera_order: [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
               CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT]
context_indices: [0]
novel_indices: [1, 2]
append_context_to_targets: true
num_context_timesteps: 1
num_target_timesteps: 1
num_max_cameras: 6
timespan: 2.0                 # 保留非零归一化常数；实际时间恒为 0
static_scene: true           # 新增：拒绝真实时间并绕过位移计算
load_depth: true             # 目标 Metric3D，不是 LiDAR
load_flow: false
load_ground: false
skip_sky_mask: true
use_dynamic_mask: true
dynamic_mask_scope: novel_12 # 已确认：末尾 6 路输入视角目标全有效
load_rel_depth_train: false
load_rel_depth_eval: true
relative_depth_source: da2
subset_ratio: 1.0
val_selection: {stop: 30000, stride: 3000, limit: 10}
mini_selection: {offset: 0, stride: 14, limit: 2048}
test_selection: all
```

`timespan=2.0` 只是原接口所需常数，不能设置为零而引起归一化除零；`context_time` 与 `target_time` 均为零，位移还要在渲染器内显式关闭。目标 18 路并不意味着 `num_max_cameras=18`：物理相机始终是 6 个。

### 2.3 公共实验配置与分辨率配置

```yaml
# configs/experiment/omniscene_base.yaml
dataset_config: ../dataset/omniscene.yaml
model: STORM-B/8
gs_dim: 3
decoder_type: dummy
num_cams: 6                  # 需实际传入 STORM 构造函数
num_motion_tokens: 16
use_sky_token: true
use_affine_token: true
use_latest_gsplat: false
near: 0.2
far: 400.0
scale_offset: -2.3
opacity_offset: -2.0
max_scale: 0.5
tau: 0.5
projected_motion_dim: 32
disable_pos_embed: false
sigmoid_rgb: false           # 保留本项目当前实现
disable_grad_checkpointing: false
enable_depth_loss: true
enable_flow_reg_loss: true
flow_reg_coeff: 0.005
enable_perceptual_loss: true
perceptual_weight: 0.05
perceptual_loss_start_iter: 5000
enable_sky_depth_loss: false
enable_sky_opacity_loss: false
lr: 0.0004
lr_sched: cosine
min_lr: 0.0
warmup_iters: 5000
weight_decay: 0.05
grad_clip: 3.0
batch_size: 1
eval_batch_size: 1
test_batch_size: 1
num_iterations: 100001
val_every_n_iters: 1000
test_every_n_validations: 10
test_at_training_end: true
ckpt_every_n_iters: 5000
keep_n_ckpts: 1
save_final_checkpoint: true
log_every_n_iters: 50
vis_every_n_iters: 5000
num_vis_samples: 1
eval_view_groups: [all_18, novel_12]
pcc_depth_mode: accumulated_z
train_depth_mode: expected_z
eval_radius_clip: 0.0
eval_render_chunk_size: 6
timing_warmup_samples: 5
num_workers: 16
seed: 1
precision: bf16
load_from: null
resume_from: null
auto_resume: true            # 默认自动恢复当前实验目录
enable_wandb: true
wandb_mode: offline
enable_feishu: true
feishu_module_paths: [~/Libraries, /vepfs-mlp2/c20250502/haoce/dzp]
project: omniscene
output_dir: ./work_dirs
```

AdamW 的 `betas=(0.9,0.95)`、无梯度累积、FP32 主参数及原生 `NativeScaler` 保留代码行为。权重衰减沿用 `timm.optim.optim_factory.param_groups_weight_decay` 的参数分组，不能把 bias/归一化参数全部强行加入衰减。`lr` 是绝对学习率，不因 batch size=1 按 `blr * batch/256` 再缩小。

`eval_radius_clip=0.0` 对齐原训练/评估调用 `model(input_dict)` 的行为。重用 `from_gs_params_to_output` 时必须去除其写死的 `radius_clip=4.0`，否则拆分接口会改变指标。渲染分块仅控制目标渲染显存，不能改变输入、目标集合或重建次数。

```yaml
# configs/experiment/omniscene_112x200.yaml
extends: omniscene_base.yaml
input_size: [112, 200]
exp_name: STORM-B-8_omniscene_112x200
```

```yaml
# configs/experiment/omniscene_224x400.yaml
extends: omniscene_base.yaml
input_size: [224, 400]
exp_name: STORM-B-8_omniscene_224x400
```

### 2.4 编码器、输出及辅助模块的具体参数

| 模块 | 设置与含义 |
| --- | --- |
| 输入通道 | 9：RGB 3 + Plücker ray 6；不拼接 Metric3D、DA2 或 mask |
| Patch embedding | patch/stride=8；两种分辨率均无需裁剪或 padding |
| Transformer | hidden dim 768，12 层，12 heads，MLP ratio 4，保留原位置编码和 gradient checkpointing |
| 时间 embedding | 保留原 TimestepEmbedder，256 维频率编码、768 维输出；只接收零，零输入不等于移除该模块 |
| GS head | Linear `768 → 8²×12`，unpatch 后每输入像素一个高斯；12 通道为深度 1、尺度 3、旋转 4、不透明度 1、颜色 3 |
| 几何激活 | `z=0.2+sigmoid(raw_z)×(400−0.2)`；`scale=min(exp(raw_scale−2.3),0.5)`；`opacity=sigmoid(raw_opacity−2)`；沿相机射线生成中心 |
| 颜色/decoder | 3 维直接颜色，保留 `sigmoid_rgb=false`；DummyDecoder，无 Latent-STORM 特征图解码器 |
| Motion | 16 个 tokens；原 8 倍上采样通道 `768→512→256→128`，投影维度 32，温度 0.5；预测及正则仍执行，预测速度不移动高斯 |
| Sky | 1 个 768 维 token，保留原 `ModulatedLinearLayer(3,512,768,3)`；只由输入及 RGB 损失学习 |
| Affine | 6 个 768 维 tokens；原 `Linear(768,12)`；目标渲染按物理 camera ID 选对应结果 |
| Renderer | 使用本项目固定 gsplat 版本；训练 RGB+ED，PCC 评估取 D；near/far 用本项目的 0.2/400 米 |

112×200 每相机 350 个 patch，六相机加辅助 tokens 共 2,123 个；224×400 每相机 1,400 个 patch，共 8,423 个。初始高斯数分别为 `6×112×200=134,400` 和 `6×224×400=537,600`，本方案不加新的裁剪、剪枝、稠密化或 top-k 预算。

两种分辨率均已完成单 bin 的前向、全部损失反向与评估，短时显存结果见第 10 节。保留模型、输入和训练 batch 的定义，使用 checkpointing 与目标渲染分块；单 bin 结果不代表整个数据集或长期训练的最大显存。

## 3. 真值、损失与训练节奏

### 3.1 损失定义

每步从六路 RGB 重建一次，渲染全部 18 路目标后计算：

\[
L=L_{RGB}+\mathbb{1}_{i\ge5000}\,0.05L_{LPIPS}+L_{depth}+0.005L_{flow}.
\]

这里 `i` 是从零开始的 optimizer 更新索引。RGB、深度与 flow 的损失类型及权重取自本项目；动态掩码是适配所增加的有效区域约束。

- **RGB 真值**：18 路原图。保持本项目 `MEAN=STD=[0.5,0.5,0.5]` 的编码器归一化；损失侧还原到 `[0,1]`。MSE 权重 1，在有效 RGB 元素上求均值：`sum(M*(pred-gt)^2)/(3*sum(M))`。不把被屏蔽区域计入分母。
- **LPIPS**：继续用 `storm/utils/lpips_loss.py` 和本项目自带的 VGG LPIPS，实现及输入数值约定保持原样；第 5,001 次更新（`i=5000`）起权重 0.05。动态区域在预测与真值的 `[0,1]` 副本上同时置零，再计算 LPIPS；不得原地改写供其他损失/指标使用的张量。它是遮挡后的图像感知损失，并非严格的逐有效像素 LPIPS。训练 LPIPS 不替换为 depthsplat 的损失类。
- **深度真值**：18 路已有 Metric3D，单位米，使用原生 `compute_depth_loss` 的归一化 L1 形式。令 `V=isfinite(Dgt) & (Dgt>0.01) & M`，`Dmax` 为该样本 18 路有限正深度的最大值，则 `Ldepth=mean_V(abs(Dpred/Dmax-Dgt/Dmax))`，权重 1。`Dpred` 是 renderer 的 expected z-depth；不改为 disparity loss，不做预测/真值的仿射对齐，不引入 LiDAR 或置信度筛选。非有限值不得参与 `Dmax`。
- **Flow 正则**：`mean(forward_flow²)`，系数 0.005，保持原模型正则；没有光流/场景流真值。静态渲染分支不使用这些预测速度。即使关闭 motion segmentation 的可视化输出，也不能因 `flow_key=None` 而意外跳过该正则，应直接从高斯预测结果计算。
- **关闭项**：sky depth、sky opacity、sky flow 以及依赖天空掩码的附带 opacity 损失全部关闭；DummyDecoder 没有额外 decoder depth loss。

掩码语义统一为 **True/白色=有效，False/黑色=排除**。参考项目中名为 `valid_depth_mask` 的变量有一次取反，不能按变量名照搬。全无效样本、无有效深度及非有限损失必须显式报错并记录 bin，不能返回 NaN 或静默退回全图监督。

### 3.2 学习率与计步

沿用 `storm/utils/misc.py::adjust_learning_rate`：令 `N=100001`，`W=5000`，`lr_max=4e-4`，`lr_min=0`。

```text
i < W:  lr(i) = lr_max * i / W
i >= W: lr(i) = lr_min + (lr_max-lr_min)/2
                   * (1 + cos(pi*(i-W)/(N-W)))
```

因此配置值 `0.0004` 是 warmup 后的峰值，首个更新按原实现使用零学习率。循环执行 `i=0...100000`，用 `completed_steps=i+1` 触发验证、测试和保存。不要把原循环里 `>`、日志迭代器的停止条件和 checkpoint 的零基编号混用而多跑/少跑一步。

| 事件 | 触发时刻与范围 |
| --- | --- |
| 训练 | 训练清单全部 bin，沿用随机/无限采样；不按 epoch 近似控制进度 |
| 常规验证 | completed_steps=1,000、2,000、…、100,000；固定 10 个验证 bin；输出两个视角组的四项指标 |
| 训练中 mini 测试 | 第 10、20、…、100 次验证后，即每 10,000 步；固定 2,048 个 bin |
| 最终 mini 测试 | 完成 **100,001** 步后必做；不能以刚做过 100,000 步测试为由跳过 |
| Checkpoint | 每 5,000 步及最终 100,001 步；周期文件按原保留策略，最终文件单独保护 |
| 完整测试 | 独立 test 命令，默认 30,080 个 bin；不得把 mini 结果标为完整测试结果 |

原项目已有周期 evaluate 和训练后 evaluate，但没有独立“每 N 次验证测试”的计数器。适配时新增 `val` 与 `mini_test` 两条评估路径，仍由本项目循环调用。可视化也改用静态目标协议，不调用要求真实时间/天空/场景流的原逻辑。

断点恢复要保存模型、optimizer、scaler、已完成步数、验证次数、已完成的 mini 测试步数、随机状态与采样状态。若最终 checkpoint 已保存但最终 mini 测试未完成，恢复时补做最终测试后才标记训练完成。离线 W&B 不执行 sync；LPIPS/VGG 的已有权重在启动前检查，缺失时报告，不在训练途中联网下载。

自动续训默认启用，重新执行相同配置的训练命令即可。仅在当前实验的 `output_dir/project/exp_name/checkpoints/`（默认位于 `work_dirs`）内选择 checkpoint，显式 `resume_from` 优先；否则优先 `ckpt_final.pth`，再选步数最大的 `ckpt_step_XXXXXX.pth`。忽略尚未写完的 `.pth.tmp` 文件，不扫描其他实验或另一种分辨率的目录。成功加载后先校验训练协议、训练/评估清单及采样游标，恢复模型、optimizer、scaler、Python/NumPy/PyTorch 随机状态；按恢复步数继续 cosine/warmup，并先补做该 checkpoint 待完成的评估。自动选择的路径和恢复步数记录在日志与 `provenance.json` 中。

无 checkpoint 时从头训练；已有 checkpoint 若损坏或配置不兼容则报错，不静默覆盖为新实验。训练和最终 mini 均完成时提示完成并返回，不重复训练、评估、飞书通知，也不重写最终 checkpoint。`auto_resume` 只作用于训练，不妨碍独立测试的 `--load_from`。可用 `--no-auto_resume` 关闭，但已有 checkpoint 的目录仍有防覆盖检查；要重新从头跑请使用新的 `exp_name`。这是重启命令后的自动恢复，不负责自动重启被中断的进程；仍按原配置每 5,000 步及最终步保存，未成功落盘的进度需重算。

### 3.3 飞书关键日志推送

按新增需求，复用 SVF-GS `trainer.py` 中的 `from auto_monitor.send_feishu import send_feishu`，调用 `send_feishu(title, body)`。`storm/utils/feishu.py` 负责加载和组织消息，训练入口与独立 mini 测试入口负责触发。两个分辨率继承公共配置，默认 `enable_feishu: true`；无需额外 export。模块搜索路径 `feishu_module_paths` 默认与 SVF-GS 一致，包含 `~/Libraries` 和 `/vepfs-mlp2/c20250502/haoce/dzp`，部署到其他环境时可在配置中修改。

| 触发点 | 推送内容 |
| --- | --- |
| 训练初始化完成、首次更新之前 | 实验名、工作目录、模型/分辨率、已完成/总迭代、初始化或恢复来源、模型三类参数量、batch size、峰值学习率及调度、验证/mini 间隔、W&B 模式 |
| 每次训练中 mini 测试成功完成 | 完成步数、实际/预期 bin 数、`all_18` 与 `novel_12` 的 PSNR/SSIM/LPIPS/PCC、模型参数量、完整高斯重建均值耗时及计时有效性、当前运行最近一步损失/学习率/梯度范数、运行耗时及剩余时间估计、checkpoint 和评估输出目录 |
| 第 100,001 步最终 mini 或恢复后补做最终 mini | 同上，标题标明“最终 mini 测试完成” |
| 独立 `--mode test --test_split mini` 成功完成 | 两组指标、覆盖范围、参数量及重建耗时、checkpoint 和输出目录；没有训练损失或训练剩余时间 |

常规 `val`、独立 `total`、`--dry_run`、`--mode check-data` 不推送；已完成全部训练及最终 mini 的 checkpoint 再次恢复不会重复推送。测试报错时不发“完成”消息；显式 `--max_eval_bins` 截断时标题标注“调试截断”，同时列出未截断集合大小。共享 GPU 或预热后无样本时不把计时称为有效独占 GPU 测量。剩余时间按本次启动以来每个更新的均摊耗时估算，包含验证/mini 的时间；刚恢复且尚未更新时显示“待估算”。

接收目标由外部模块已有的 `~/.feishu_env`（`FEISHU_WEBHOOK_URL`）决定，项目不复制 webhook 到 YAML、checkpoint 或日志。外部模块依赖的 `requests`、`python-dotenv` 已列入 requirements。当前本地模块的 HTTP 请求 timeout 为 5 秒；模块缺失、依赖/配置缺失、网络错误、返回失败均只记录 warning，不终止训练，也不自动重试。本地 `feishu_notifications.jsonl` 保存事件、步数、标题、正文和发送状态；外部模块可能含 webhook 的错误输出不写入实验日志。通知开关和模块路径不参与 checkpoint 的训练协议签名，可以在恢复时更改。调试时使用 `--no-enable_feishu` 关闭。

## 4. OmniScene 数据加载及与 depthsplat 的差异

### 4.1 清单、路径与抽样

两个参考项目的数据软链接当前都指向 `/home/B_UserData/dongzhipeng/Datasets/dataset_omniscene`。本次只读核对得到：训练清单 135,932 个 bin，评估清单 30,080 个 bin。还检查了一个 bin 中前三个 CAM_FRONT 条目的 RGB、参数、Metric3D、DA2 和 mask 资产；这不是对全部数据的完整性证明。

按 JSON 中的原有顺序选择，禁止排序后再切片：

```python
train_bins = train_manifest["bins"]
val_bins = eval_manifest["bins"][:30000:3000][:10]
mini_bins = eval_manifest["bins"][0::14][:2048]
test_bins = eval_manifest["bins"]
```

验证和 mini 都是同一评估清单的子集，是沿用已有项目的监控协议，不另行声称拥有独立验证划分。不要改成 `center150` 或其他历史子集；每次评估保存实际 bin 清单及哈希，核验唯一性、预期条数和完成条数。

资产路径沿用参考 loader 的映射：

| 数据 | `samples` 路径映射；`sweeps` 同理 |
| --- | --- |
| RGB | `samples_small/...jpg`，已有 224×400 图像 |
| 内参 | `samples_param_small/...json` 中的 `camera_intrinsic` |
| Metric3D | `samples_dptm_small/..._dpt.npy` |
| DA2 | `samples_dpt_small/...npy` |
| 动态掩码 | `samples_mask_small/...png` |

从相应 bin 的 pickle 元信息读取 `sensor_info` 中的相机记录。只读取六个 camera 列表，不沿用 depthsplat 中依赖 `LIDAR_TOP` 获取帧数的代码；不打开 LiDAR 文件。相机记录里的 `sensor2lidar_transform` 是**已给定相机外参所使用的参考坐标系**，使用这个矩阵不等于使用 LiDAR 点云。

### 4.2 固定视角顺序

物理相机顺序固定为：

```text
0 CAM_FRONT
1 CAM_FRONT_RIGHT
2 CAM_FRONT_LEFT
3 CAM_BACK
4 CAM_BACK_LEFT
5 CAM_BACK_RIGHT
```

每个相机的索引 0 是输入；索引 1、2 是两个新视角目标。目标按相机依次添加两张，最后添加输入六张：

```text
context = [cam0[0], cam1[0], ..., cam5[0]]
target  = [cam0[1], cam0[2], cam1[1], cam1[2], ..., cam5[1], cam5[2],
           cam0[0], cam1[0], ..., cam5[0]]
target_camera_ids = [0,0,1,1,2,2,3,3,4,4,5,5,0,1,2,3,4,5]
all_18  = target[0:18]
novel_12 = target[0:12]
```

`novel_12` 是 `all_18` 的子集；每个 bin 仅重建一次。不能将这份相机优先排列的列表直接 reshape 成三个时刻、六个相机，也不能按“时间与输入相同”过滤目标。

### 4.3 预处理、内外参与掩码

- **RGB**：先读取已有 224×400 资产；低分辨率在线 resize 为 112×200，高分辨率保持原样。沿用 depthsplat 的 PIL RGB resize 行为并固定库版本/插值约定，不额外做随机 crop、翻转或颜色增强。loader 输出 `[0,1]`，送入 STORM 时转换为 `[-1,1]`。
- **内参**：从对应 small 参数文件读取，在 resize 时按宽高分别缩放 `fx,cx` 和 `fy,cy`。STORM 需要像素单位 K；depthsplat 输出的是第一行除 W、第二行除 H 的归一化 K。因此移植时直接保留像素 K，或在接口处恰好反归一化一次；禁止拿原始全分辨率 K 与 small 图像配对。
- **外参**：沿用 depthsplat 的 OpenCV `c2w=sensor2lidar_transform`，六输入和十八目标放在共同参考坐标系，单位米。渲染器使用 `w2c=inverse(c2w)`；不复制 utils 中另行拼装的旧转置矩阵。不要复制 SVF-GS 为其 OpenGL 渲染器所做的 `flip_yz`，也不要重复应用 STORM 原生 Waymo 数据集的坐标变换。
- **Metric3D**：读 `*_dpt.npy` 并转 FP32，按目标分辨率 bilinear resize，保留米制值，不除以 1,000，不做 min-max 或 unit-baseline 归一化。不读取 `*_conf.npy`，不导入 SVF-GS 的置信度阈值。
- **掩码**：沿用参考实现的灰度读取、bilinear resize、除 255 后转 bool（大于零为有效）；不擅自换成另一阈值或 nearest。`dynamic_mask_scope=novel_12`，只读取十二路新视角真实掩码，最后六路全有效。掩码不把输入 RGB 的动态区域抹掉，只控制监督损失。
- **DA2**：只在评估 loader 读取；处理顺序见第 6 节。

### 4.4 能否直接复用 depthsplat 的加载方式？

**可以移植其选 bin、选视角、路径映射和图像/相对深度预处理；不能原封不动接入。**

| 接口 | depthsplat 当前方式 | 本项目所需改动 |
| --- | --- | --- |
| 数据组织 | `context`、`target` 嵌套字典 | loader 可保留此契约，再用专用 adapter 转 STORM 输入 |
| 内参 | 归一化 K | 像素 K |
| 图像数值 | `[0,1]` | 模型前归一化 `[-1,1]`，损失/评估还原 |
| 时间维 | loader 无 STORM 的显式时间维 | 插入长度 1 的虚拟维；时间张量全零 |
| 深度监督 | 当前 OmniScene loader 没加载 Metric3D | 从 SVF-GS 的路径规则移植 18 路 metric depth 加载 |
| PCC | 主要在 test 路径读取 DA2 | val、mini_test、test 三条评估路径均可读取 |
| 数据 shim | encoder 内部按 patch 对齐裁剪 | 不调用 depthsplat shim，保持用户指定完整分辨率 |
| 框架依赖 | Hydra、Lightning、typed config | 本项目 Dataset/DataLoader 与 argparse 配置层 |

**分辨率差异必须在比较结果中标明。** depthsplat 当前默认 shim 要求 16 整除，会将 112×200 中心裁成 **112×192**；224×400 不受影响。STORM-B/8 的 patch size=8，两种指定分辨率都可直接使用。本文不修改 depthsplat，也不把 STORM 裁成 112×192。历史低分辨率结果应标注实际输入/评估尺寸；要求像素范围完全相同的比较时，需要另行审阅 depthsplat 的对齐方案。

## 5. 主程序的数据接口与调用

### 5.1 专用 adapter 的张量契约

batch size B=1，T=1 为虚拟时间维：

| 字段 | 形状/用途 |
| --- | --- |
| `context_image` | `[B,1,6,3,H,W]`，唯一送入图像编码器的图像；adapter 保留 `[0,1]`，`reconstruct_static` 内归一化到 `[-1,1]`，计时包含此操作 |
| `context_intrinsics` / `context_camtoworlds` | `[B,1,6,3,3]` / `[B,1,6,4,4]` |
| `context_time` | `[B,1]` 全零，不携带每相机真实时间 |
| `target_intrinsics` / `target_camtoworlds` | `[B,1,18,3,3]` / `[B,1,18,4,4]`，仅供渲染 |
| `target_time` | `[B,1]` 全零 |
| `target_camera_ids` | `[B,1,18]`，映射六个 affine 结果 |
| `target_image` | `[B,1,18,3,H,W]`，监督字典 |
| `target_depth` / `target_valid_mask` | `[B,1,18,H,W]`，监督字典 |
| `target_rel_depth` | `[B,1,18,H,W]`，只存在于评估指标字典 |

模型重建函数只接收白名单中的六路图像与相机参数、常零时间及尺寸信息。目标 RGB、Metric3D、DA2、掩码不传入重建函数；目标相机参数只在后续渲染阶段使用。

### 5.2 调用流程及必须修改的假设

```text
读取配置 → OmniScene Dataset/DataLoader → 专用 adapter
                                      ├─ context → 重建高斯、motion/sky/affine 状态
                                      └─ target cameras → 静态渲染全部 18 路
训练：监督字典 → masked RGB / STORM LPIPS / Metric3D / flow 正则
评估：指标字典 → all_18 / novel_12 → 逐 bin 记录 → 数据集汇总
```

可以借鉴 depthsplat 的“encoder 重建 → decoder 渲染 → loss/metrics”组织方式，但不能直接调用其 `ModelWrapper` 或 `encoder/decoder` 接口。本项目 `STORM.forward` 已合并特征提取、运动预测、渲染、天空及颜色处理；现有 `get_gs_params` 与 `from_gs_params_to_output` 可作为拆分起点。

需要逐项适配：

1. 原 `prepare_inputs_and_targets` 带三相机及时间采样假设，新增 OmniScene adapter；不用原生 `depth_flows`/sky 数据路径拼一个伪数据集。
2. 原入口只传 `num_motion_tokens` 等参数，**没有把 `num_max_cameras` 传给构造函数的 `num_cams`**。必须显式传 `num_cams=6`，否则 affine token 仍为 3 个。
3. 编码器输入相机数 6 与目标视图数 18 分开处理；渲染 reshape 使用 `tgt_t/tgt_v/tgt_h/tgt_w`，不能沿用输入 v/H/W。每个目标通过 camera ID 获取 affine，不能简单复制三组或对六相机取均值。
4. `static_scene=true` 时断言时间为零，并直接用原高斯中心渲染，绕过 `forward_flow * Δt` 位移分支。运动模块保留，不将 `num_motion_tokens` 改成零；原模型零 token 会转入直接速度预测分支。
5. 原 evaluate 按 context frame index 剔除目标帧；全时间零时会错误排除目标。新增 OmniScene 评估器，固定按索引取 `all_18`/`novel_12`，关闭 flow/真实时序评估。
6. 将两条模型调用路径共用同一重建与渲染实现，统一 radius clip、天空合成、原有 affine 运算和 dummy decoder。验证拆分前后输出等价，再用于计时。天空背景仅参与 RGB，不给 PCC 深度人为填 300 米，也不通过 affine 改变深度。

原始 affine 应用和射线像素中心实现有值得独立核查的细节：当前 affine 使用源码中的 einsum，并非可直接假设为标准 `A×RGB+b`；Plücker ray 网格构造及后续坐标各有一次 `+0.5`。本次适配保留原算法行为，几何检查需记录这些原生约定，不夹带未审阅的算法修正。若后续要修正，另列实验变更。

### 5.3 训练与测试命令

以下是正式训练/完整测试命令，目前仅完成冒烟验证，未启动这些正式任务：

```bash
python main_storm.py --config configs/experiment/omniscene_112x200.yaml --mode train
python main_storm.py --config configs/experiment/omniscene_224x400.yaml --mode train

python main_storm.py --config configs/experiment/omniscene_112x200.yaml --mode test --test_split total --load_from work_dirs/omniscene/STORM-B-8_omniscene_112x200/checkpoints/ckpt_final.pth
python main_storm.py --config configs/experiment/omniscene_224x400.yaml --mode test --test_split total --load_from work_dirs/omniscene/STORM-B-8_omniscene_224x400/checkpoints/ckpt_final.pth
```

新增 `--mode {train,test}`、`--test_split {mini,total}`；训练中的评估由循环调度。`load_from` 仅加载模型用于测试/明确初始化，`resume_from` 用于恢复完整训练状态，二者不能混用。完整测试默认不保存所有图片或视频，避免 I/O 占用；保留指定少量可视化和全部数值记录。

## 6. PCC 与另外三项指标

### 6.1 DA2 参考深度

直接移植 depthsplat 的 `load_conditions` 中 DA2 转换，并固定操作顺序：

```python
disp = load_existing_da2_npy().astype(float32)
disp = bilinear_resize_if_needed(disp, (H, W))
ratio = min(disp.max() / (disp.min() + 0.001), 50.0)
lower = disp.max() / ratio
relative_depth = 1.0 / maximum(disp, lower)
relative_depth = (relative_depth - relative_depth.min()) / (
    relative_depth.max() - relative_depth.min()
)
```

这是每幅图先由 disparity 转深度再归一化，不是直接拿原始 disparity 与渲染深度求 PCC。不能先归一化再 resize，也不能仅因为 PCC 对单个向量的仿射变换不变，就删掉每视图归一化：本协议最终会把多个视图拼接计算。

对缺失文件、非有限值、无效 ratio、常量参考图等情况记录 bin/相机/路径并报告；不静默填零，不调用 DA2 模型补算。不通过删除失败样本提高统计分数。

### 6.2 训练 ED 与评估 D 的区别

设某目标像素的高斯权重为 `w_i=α_i∏_{j<i}(1−α_j)`，`z_i` 为相机坐标系 z，累积透明度 `A=Σw_i`：

\[
D_{acc}=\sum_i w_i z_i,\qquad D_{exp}=D_{acc}/\max(A,10^{-10}).
\]

- STORM 当前 `render_mode="RGB+ED"` 返回 `Dexp`，**训练深度损失保留此定义**。
- depthsplat 的 `render_depth_cuda` 将相机 z 作为颜色、黑背景栅格化，返回 `Dacc`；SVF-GS CUDA 深度也是累积定义。
- **评估 PCC 使用 `Dacc`**。优先在相同渲染调用中用 `RGB+D` 直接取累积深度；若验证阶段还要记录训练式深度损失，可用同次 alpha 恢复 ED。另一等价实现是保留 ED 并乘 `A.clamp(min=1e-10)`；实现时用数值检查确认与直接 D 一致。

不能使用欧氏射线距离，不能忘记透明度因子，也不能对每视图的预测深度额外做 min-max 或尺度拟合。`Dacc` 与 `Dexp` 的区别是逐像素缩放，会改变 PCC，不能以 PCC 的尺度不变性忽略。

### 6.3 计算位置、分组和平均方式

在全部 18 路 RGB、深度和 alpha 按目标顺序收齐后计算；分块渲染不能分块计算 PCC 再平均。每个 bin：

1. 对每张目标 RGB 计算 PSNR、SSIM、LPIPS，然后分别在 18 张和前 12 张上平均。
2. `PCC_all_18 = Pearson(flatten(DA2[0:18]), flatten(Dacc[0:18]))`。
3. `PCC_novel_12 = Pearson(flatten(DA2[0:12]), flatten(Dacc[0:12]))`。
4. 对所有 bin 的组内分数做算术平均。batch size=1，一个 bin 就是一条场景记录。

PCC 不是“每幅图 PCC 的平均”，不是“整个数据集所有像素一次 Pearson”，也不能从 all_18 分数推算 novel_12。Pearson 使用中心化协方差除以两个标准差的乘积；可复用 depthsplat 的函数形式，但采用无状态实现或每组独立 reset，避免缓存 Metric 的跨样本状态被误用。

所有主指标均使用完整图像范围，不以训练动态掩码筛像素；不引入其他数据集的 ego mask。深度背景的零值保留，不因 alpha 小就额外过滤。若出现非有限值、有效元素不足或任一向量方差为零，标记该记录及总体异常，正式汇总不能静默忽略后声称完整覆盖。

### 6.4 PSNR / SSIM / LPIPS 对齐

复用 depthsplat/SVF-GS 的评估定义，不能沿用 STORM 原评估器不同的 SSIM 默认参数：

| 指标 | 约定 |
| --- | --- |
| PSNR | GT/预测 clip 到 `[0,1]`，每张图全通道 MSE，再 `−10log10(MSE)` |
| SSIM | skimage，`win_size=11`，`gaussian_weights=true`，`channel_axis=0`，`data_range=1.0`；固定相同版本及默认 sigma/covariance 设置 |
| LPIPS | 官方 `lpips.LPIPS(net="vgg")`，eval/frozen，RGB `[0,1]`，`normalize=true` |
| PCC | DA2 相对深度对累积 z 深度；每 bin、每组展平求相关 |

图像指标评估前统一将渲染 RGB 还原并 clip 到 `[0,1]`，与 depthsplat 的测试图像处理一致。训练自带 LPIPS 与官方评估 LPIPS 分开实例化，不用一个网络的预处理替代另一个。

### 6.5 输出与复用边界

建议每次评估输出至 `eval/{val|mini|total}/step_{completed_steps}/`：

```text
selected_bins.json       # 精确评估清单、来源哈希
records.jsonl           # 每 bin 两条记录：all_18 / novel_12，四指标
summary.json            # 两组均值、期望/完成/异常条数及协议元信息
parameters.json         # trainable / frozen / total 原始整数
reconstruction_time.json # 逐 bin 耗时及汇总
```

`summary.json` 至少记录 `reference=da2`、`depth_mode=accumulated_z`、分辨率（请求值与实际值）、视角顺序、掩码范围、split、checkpoint SHA/路径、代码 SHA。保留兼容的 `scores_pcc_all.json` 与 `scores_all_avg.json` 时，将其明确限定为 all_18，另外输出 novel_12 的文件，不能让旧字段悄然改义。

**PCC 计算公式、DA2 转换与逐 bin 平均可以复用，完整统计调用不能直接照搬。** depthsplat 普通 OmniScene 路径主要汇总 all_18；新增双组时可参考 SVF-GS 的分组统计。depthsplat 新的跨数据集评估代码含 Metric3D/PandaSet/DDAD 元信息，不得把这些 reference、ego mask 或 temporal18 标记复制成 OmniScene 的实际协议。

## 7. 参数量与完整重建耗时

### 7.1 参数量

构造每种分辨率的最终模型、加载 checkpoint 后，按唯一 Parameter 对象去重计数：

```text
trainable = sum(numel(p) for unique model parameters if p.requires_grad)
frozen    = sum(numel(p) for unique model parameters if not p.requires_grad)
total     = trainable + frozen
```

范围包括重建/渲染模型使用的 encoder、GS head、时间/运动、sky、affine 和 decoder 的参数；被冻结但仍用于模型前向的参数也计入。LPIPS 训练损失网络与指标网络不属于重建模型，单独注明且不混入 total；buffer、optimizer 状态不算参数。

本实验保留的时间/运动模块全部计数，不能以其不产生位移为由剔除。两个分辨率须分别实际统计：可学习位置编码尺寸不同，不能直接引用论文里的 Base 参数量；位置编码自身的差值为 `(1400−350)×768=806,400` 个参数，最终总数以实例为准。

### 7.2 计时边界

主报告单位为 **ms/bin（六路输入完整重建一次）**：数据已在 GPU、模型已加载，开始计时；包含模型侧 RGB 归一化、ray/Plücker 构造、编码器、全部高斯参数预测与激活、保留的运动预测、sky 条件 token 和 affine 参数生成；获得可供任意目标渲染的完整场景表示后结束。

不计磁盘读取、DataLoader 等待、CPU→GPU 拷贝、模型初始化、最终目标图像渲染、损失、指标与文件保存。这里“完整”意味着不能只计 Transformer backbone；若后续重建内部增加必要处理，其耗时也必须包含。目标天空颜色是在目标渲染阶段根据 ray 和 sky token 生成，不属于重建输出图像。

计时前后均 `torch.cuda.synchronize()`，使用 `time.perf_counter()`；不能直接复制 depthsplat 只有 `time.time()`、未同步 GPU 的计时器。`eval()`、`inference_mode()`、固定推理精度与训练配置兼容；每个 bin 只重建一次，两个视角组共用结果。不得将耗时除以 18 或 6 后标为完整重建时间。

前 5 个样本用于运行时预热，仅从计时统计中排除，质量指标仍包含它们。保存每个 bin 的时间、有效计时数量、均值/中位数/P90，以及 GPU、PyTorch/CUDA/gsplat 版本、dtype、分辨率、重建高斯数。mini 与完整测试的耗时结果分别标明范围，不混为一个表项。目标渲染耗时可另列辅助字段。

## 8. 实现文件及验收

### 8.1 实现位置

| 文件/目录 | 职责 |
| --- | --- |
| `main_storm.py`、`storm/omniscene_config.py`、`storm/omniscene_runner.py` | YAML/CLI 合并；OmniScene 分支；显式模型参数；验证/mini/最终测试调度；离线日志与恢复 |
| `configs/dataset/`、`configs/experiment/` | 上述数据与两个分辨率配置 |
| `storm/dataset/omniscene_dataset.py`（新增） | bin 清单、6→18 视角、RGB/K/pose、Metric3D、mask、评估 DA2 |
| `storm/dataset/omniscene_adapter.py`（新增） | 归一化、像素内参、单虚拟时间维、重建/监督/指标字典隔离 |
| `storm/models/storm.py` | 静态位移开关；6/18 维度；camera ID affine；拆分共用重建/渲染；ED/D 输出 |
| `storm/utils/losses.py`、`storm/utils/lpips_loss.py` | 有效区域损失；保持 STORM 原损失类型/权重，保留 flow 正则 |
| `storm/evaluation/omniscene.py`（新增） | 两组四指标、DA2 PCC、覆盖检查、逐 bin 与汇总输出 |
| `storm/utils/` 的独立统计工具 | 参数计数、同步重建计时及结果元信息 |
| 日志/可视化相关调用 | W&B offline、静态目标可视化、禁用场景流及天空掩码依赖 |
| `storm/utils/feishu.py` | 复用外部 send_feishu，训练启动与 mini 结果推送、失败隔离和本地发送状态 |

只在本项目移植必要代码，并保留来源说明；不改 SVF-GS/depthsplat，不建立依赖相邻目录存在的 Python import。

### 8.2 验收内容及执行顺序

1. **配置与数据 CPU 检查**：两份配置完整展开；抽同一个真实 bin 与参考 loader 比对视角、相机、RGB/K、DA2/mask；比对发生在 depthsplat crop shim 之前。检查 Metric3D 路径/单位及 18 路对齐。正式训练前检查所需资产覆盖，缺失只报告，不补生成。
2. **几何与无时序检查**：检查 c2w 逆矩阵、resize K、相机 ID 与目标顺序；记录原射线约定。修改元信息时间戳不改变适配结果；静态分支改变预测速度不移动高斯；替换目标 RGB/深度不影响重建高斯。
3. **损失和 PCC 检查**：全有效 mask 时与原 STORM 损失一致；固定真值后，改变屏蔽像素的预测不影响相应损失；DA2 处理、两组 Pearson 与参考实现一致；验证分块拼接后计算；D 与 `ED×alpha.clamp(min=1e-10)` 在同一栅格化结果上数值一致。
4. **调度与恢复检查**：用轻量桩验证 1,000/10,000/100,001 的触发次数、最终必测、最后 checkpoint 和恢复补测；无需真的训练 100,001 步来验证调度。
5. **最小 GPU 冒烟检查**：分别对两个分辨率做 batch=1 前向、反向及 18 目标评估，检查原调用与拆分调用的静态输出、峰值显存和计时范围。本次已执行的具体检查见第 10 节。
6. **正式实验**：上述检查通过后，按两个独立配置训练；最终 mini 成功不等于完整测试完成。正式报告分别附完整清单覆盖及 all_18/novel_12、参数量、重建时间。

代码已按上述协议实现。全部资产的完整性扫描、正式训练及完整测试仍待执行；本次单步冒烟的输出只能证明相应代码路径可运行，不能作为模型效果或速度结论。

## 9. 核对来源

本方案的实现参数以当前源码为准。相关原论文入口见项目 [README](../README.md) 中的 STORM 论文链接。

- 本项目：[入口与训练循环](../main_storm.py)、[STORM 模型](../storm/models/storm.py)、[原数据适配](../storm/dataset/data_utils.py)、[损失](../storm/utils/losses.py)、[训练 LPIPS](../storm/utils/lpips_loss.py)、[原评估](../engine_storm.py)、[学习率](../storm/utils/misc.py)。
- depthsplat：[OmniScene Dataset](../../depthsplat/src/dataset/dataset_omniscene.py)、[加载辅助函数](../../depthsplat/src/dataset/utils_omniscene.py)、[主模型调用和指标汇总](../../depthsplat/src/model/model_wrapper.py)、[评估指标](../../depthsplat/src/evaluation/metrics.py)、[深度渲染](../../depthsplat/src/model/decoder/cuda_splatting.py)、[裁剪 shim](../../depthsplat/src/dataset/shims/patch_shim.py)、[112×200 配置](../../depthsplat/config/experiment/omniscene_112x200.yaml)。
- SVF-GS：[OmniScene Dataset](../../SVF-GS/data/omniscene_dataset.py)、[Metric3D/DA2/mask 加载](../../SVF-GS/data/transforms/loading.py)、[重建、监督与双组评估](../../SVF-GS/model/omni_gs.py)、[参数计数](../../SVF-GS/tools/param_count.py)、[指标](../../SVF-GS/tools/metrics.py)。

## 10. 实现后的使用与验证记录（2026-09-25）

### 10.1 环境与执行入口

使用 `storm` 环境，已安装 `scikit-image==0.25.2`、`lpips==0.1.4`，保留原 `torch==2.3.1+cu118` 和 `torchvision==0.18.1+cu118`。运行时检查 gsplat 的安装来源，要求 README 中固定的 commit，实际版本与 commit 写入结果元信息。

```bash
conda activate storm
export PYTHONNOUSERSITE=1

# 只展开/校验配置，不初始化 CUDA
python main_storm.py --config configs/experiment/omniscene_112x200.yaml --dry_run

# 正式运行前：扫描训练和完整评估清单的必要资产，不生成任何新数据
python main_storm.py --config configs/experiment/omniscene_112x200.yaml --mode check-data
# 快速检查可加 --check_limit 2，输出会明确 complete_scan=false

# CPU 回归检查
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v

# 首次运行从头训练；中断后重跑同一命令默认自动恢复，无需额外参数
python main_storm.py --config configs/experiment/omniscene_112x200.yaml
# 配置须与 checkpoint 的训练协议一致；也可显式指定 --resume_from <checkpoint.pth>
```

从头训练、完整测试的两种分辨率命令见第 5.3 节。默认自动恢复当前实验目录中的 checkpoint；要独立重跑时换一个 `exp_name`，避免混用已有实验。

训练和评估依赖的 VGG 权重只读取本地缓存：`torch.hub.get_dir()/checkpoints/vgg16-397923af.pth`；STORM 自带 LPIPS 的校准权重默认读取 `~/.cache/torch/hub/checkpoints/vgg.pth`，也可通过 `--lpips_weights` 指定已有文件，校验原项目 MD5。缺失时启动失败并给出路径，不在训练/测试中下载。

W&B 默认读取公共配置的 `wandb_mode: offline`，启动程序自动应用该值，并以 `mode=offline` 初始化，无需用户手动 export。恢复训练时日志可开启一个新的本地离线段，optimizer、步数、采样游标及 RNG 从 checkpoint 恢复，不依赖在线 W&B resume。所有数值和状态也写入本地 JSON/JSONL。

### 10.2 受限调试及与正式实验隔离

可通过 `--gpu_memory_limit_gb` 限制当前进程的 PyTorch 显存分配；启动时检查剩余显存并至少留 1 GiB 余量。此参数不是整个设备的硬隔离，其他进程的显存仍可能变化，正式运行前应重新检查资源。

以下是一种明确缩短的**冒烟命令**，不是正式配置：

```bash
python main_storm.py --config configs/experiment/omniscene_112x200.yaml \
  --num_iterations 1 --max_eval_bins 1 --num_workers 0 \
  --perceptual_loss_start_iter 0 --no-enable_wandb --no-enable_feishu \
  --gpu_memory_limit_gb 8 --exp_name smoke_112x200
```

其中提前开启 LPIPS 是为了覆盖全部损失的反向路径；正式配置仍从零基索引 5,000 开启。原 warmup 在首步使用 lr=0，所以单步检查验证前向、梯度与 optimizer 路径，不代表模型已经学到有效重建能力。

`--max_eval_bins` 只用于显式调试，默认 0 表示不截断。截断结果进入 `limited_N` 子目录，记录 `limited=true`、实际条数和未截断条数；不能当作完整 mini/total 结果。`--num_iterations` 的调试覆盖也完整写入配置。

额外提供单 bin CUDA 数值检查：

```bash
python -m tests.check_omniscene_cuda \
  --config configs/experiment/omniscene_112x200.yaml \
  --mode test --load_from work_dirs/omniscene/smoke_112x200/checkpoints/ckpt_final.pth \
  --gpu_memory_limit_gb 8
```

### 10.3 已完成的检查与结果边界

| 检查 | 已验证范围 |
| --- | --- |
| CPU 回归 | 14 项：配置继承/覆盖与协议拒绝、相机/目标顺序、无需 LiDAR/confidence/输入视角 mask 的加载、时间无关、掩码梯度、深度有效性、双组 PCC、计时预热、静态渲染接口、100,001 步触发逻辑、真实 checkpoint 故障恢复 |
| 飞书通知 CPU 回归 | 6 项：默认配置/关闭/旧 checkpoint 兼容、延迟加载及模块路径、双组指标和调试标记、失败隔离及错误脱敏；原恢复测试补充推送触发断言。全部使用模拟发送，无真实飞书消息、无 GPU 占用 |
| 自动续训 CPU 回归 | 新增 2 项，合计 22 项通过：两个分辨率默认启用、独立测试兼容、当前实验隔离、忽略临时文件、自动选择与显式路径优先级；原故障恢复测试改为不传恢复路径，并加入随机前向，验证恢复后结果与连续训练一致，已完成实验不重写最终 checkpoint |
| 外部 send_feishu 接口 | 在 storm 环境加载本机真实模块和已有 webhook 配置，拦截 HTTP 请求后验证消息载荷、5 秒 timeout 及返回状态；已补齐 python-dotenv 1.2.3。未测试真实消息送达 |
| 真实数据 | 训练/完整评估清单各检查前 2 个 bin 的必要资产；非全量扫描 |
| 与 depthsplat 对照 | 对一个真实 bin 的全部 18 路、两个分辨率，在 crop shim 前比较；RGB、mask、DA2 相同；像素内参最大差分别约 `1.53e-5` / `3.05e-5`，来自归一化/反归一化的浮点舍入 |
| 112×200 GPU | 1 次包含 RGB、LPIPS、深度、flow 正则的反向更新；最终测试 1 个 bin，输出 all_18/novel_12 四项指标；独立 `--mode test` 的 checkpoint 加载、指标和可视化通过 |
| 224×400 GPU | 同样完成 1 次全部损失反向与最终测试 1 个 bin |
| GPU 数值 | 112×200 的一个真实 bin：同配置 ED/D 换算通过；修改预测速度不改变静态图像；六路对应目标与原生 STORM 调用的 RGB/深度/alpha 对照通过 |
| 恢复 | CPU 故障注入：在周期评估失败后恢复，更新结果与不中断训练相同；最终测试缺失时补做，完成后再次恢复不重复测试 |

本次 4090 上的短时训练峰值分配显存约为 112×200 **4.68 GiB**、224×400 **7.43 GiB**，对应进程分配上限分别为 8 / 12 GiB。112×200 的数字来自加入后续 LPIPS/sky 激活重算之前的检查；不表示低分辨率的最终最低占用。目标渲染固定分块 6，LPIPS 按视图计算并在训练时重算激活，sky head 同样启用训练激活重算，不减少 18 路监督或改变模型容量。高分辨率所测 reserved 峰值约 8.57 GiB；这些均不是长期训练上界。

实际模型参数量（不含损失与指标网络）：

| 分辨率 | 可训练参数 | 冻结参数 | 总参数 |
| --- | ---: | ---: | ---: |
| 112×200 | 100,409,010 | 1 | 100,409,011 |
| 224×400 | 101,215,410 | 1 | 101,215,411 |

已把相机矩阵求逆移到目标分块之前。在上述真实 bin 的 CUDA 检查中，分块 1 与 6 的 ED、D、alpha 结果一致；BF16 天空/affine 的矩阵计算仍可能随 batch 形状有舍入差异，因此正式实验固定 `eval_render_chunk_size=6`，并在结果中保存该值，不在不同检查点之间切换。

结果元信息记录实际依赖版本、代码工作树哈希、模型权重哈希、清单哈希、请求/实际分辨率、目标顺序、渲染深度及分块设置。`model_weights_sha256` 用于识别内存中评估的权重，不受 checkpoint 中测试完成标记/RNG 等元数据更新的影响；独立测试另外记录 checkpoint 文件 SHA256。

计时前后记录同卡其他计算进程。存在竞争进程、没有足够预热后的样本，或无法确认设备状态时，`valid_gpu_measurement=false`。本次 GPU 上已有训练任务，且冒烟只有 1 个评估 bin，所以没有报告正式重建速度。

本地冒烟输出位于 `work_dirs/omniscene/smoke_112x200/` 与 `work_dirs/omniscene/smoke_224x400/`，不纳入 Git。尚未运行：全量资产扫描、100,001 步正式训练、2,048 个 bin 的完整 mini 测试以及 30,080 个 bin 的完整测试。
