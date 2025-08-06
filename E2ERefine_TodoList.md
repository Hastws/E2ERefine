
# ✅ E2ERefine 项目全流程 TodoList

## 第一阶段：工具准备 + Python 基础

- [X] 完成 AutoAlgorama 工具库的迁移集成
- [ ] 安装 Python + pip + conda（推荐 miniconda）
- [ ] 安装 PyTorch + Jupyter Notebook + VSCode 插件
- [ ] 学习张量操作（torch.Tensor，广播，索引）
- [ ] 熟悉自动求导机制（`requires_grad`、`backward()`）
- [ ] 了解简单模型训练 loop（loss + optimizer）
- [ ] 阅读《深度学习入门》1~4 章（斋藤康毅）

## 第二阶段：C++ iLQR 求解器

- [ ] 阅读 iLQR 理论：Forward、Backward、Line Search
- [ ] 编写 Forward 推演代码（预测下一个状态）
- [ ] 实现雅可比矩阵：∂f/∂x, ∂f/∂u
- [ ] 完成 Value iteration 的 backward pass
- [ ] 写一个 toy problem 测试是否能收敛

## 第三阶段：C++ 工程模块化

- [ ] 创建 `OptimizerILQR` 类封装
- [ ] 封装 CostFunction 接口（支持 jerk/curvature）
- [ ] 添加 SoftBound 支持（超限惩罚）
- [ ] 实现轨迹输出到 CSV，便于可视化
- [ ] 添加终端成本（terminal cost）

## 第四阶段：Python 封装接口

- [ ] 学习 pybind11 或 C API 基础用法
- [ ] 暴露 `optimize(state, ref_traj)` 接口到 Python
- [ ] 写一个 Python 脚本测试输出是否正确
- [ ] 加入日志输出、debug flags 控制

## 第五阶段：端到端 E2E 模型训练

- [ ] 准备 nuScenes-mini 数据集（或 mock 数据）
- [ ] 写 `Dataset` 加载类，提取轨迹、BEV 图
- [ ] 构建 BEV+MLP 的轨迹预测模型
- [ ] 编写 imitation loss（MSE/KL），支持 best-of-K
- [ ] 开始训练 & 输出预测结果

## 第六阶段：iLQR 后处理集成

- [ ] 将 E2E 输出作为参考轨迹传入 iLQR
- [ ] 对比 E2E 原始轨迹 vs iLQR 优化后轨迹
- [ ] 可视化对比：jerk, curvature, 终点误差（ADE/FDE）
- [ ] 输出指标报告（表格 + 图）

## 第七阶段：轨迹蒸馏优化

- [ ] 离线批量生成 iLQR 优化后的教学轨迹
- [ ] 训练时添加 distill loss（E2E vs iLQR）
- [ ] 观察 loss 是否下降、指标是否提升

## 第八阶段：强化学习模块

- [ ] 构建 gym-like 环境（历史状态 + 地图）
- [ ] 设置 reward 函数： imitation + smoothness + safety
- [ ] 使用 PPO/DDPG 训练 RL policy
- [ ] 可视化训练曲线，比较 RL vs iLQR 效果

## 第九阶段：闭环测试 + 鲁棒性分析

- [ ] 构建仿真闭环流程（输入 → 模型 → 控制 → 输出）
- [ ] 注入误差、遮挡，测试鲁棒性提升效果
- [ ] 总结评估指标，整理为对比图表

## 第十阶段：展示与总结

- [ ] 写 `README.md` + 项目结构说明
- [ ] 绘制一张整体架构图（Visio 或 draw.io）
- [ ] 输出演示视频（比如轨迹对比）
- [ ] 写项目博客或文档（掘金/知乎/GitHub Pages）
- [ ] 在简历中添加这项工程经历（含关键词）
