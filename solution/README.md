# 难点与测试对应

## 难点 1：读出堆积要一起估

推理轴：数值方法

落在推理路径上的哪里：仪器读到的是 y = u/(1 + k·u)，k 是每计数上的堆积系数，记录里不给。忽略堆积（当成 k=0）时高计数段被系统性压低，快成分的时间常数会偏大、幅度会偏小；把 k 硬编码成某个值也过不了另一条曲线。k 和两个时间常数是非线性耦合的，只给定这三个量时幅度与基线才是线性的，所以要先在网格上粗定位，再在实测残差上做局部精修。原实现是单指数对数回归，k 这件事完全没进模型。

对应测试：test_pileup_coefficient_recovered、test_tau_pair_matches_independent_fit

## 难点 2：两个时间常数要分开搜

推理轴：数值方法

落在推理路径上的哪里：双指数模型对两个时间常数是非线性的，给定两个时间常数后幅度与基线才是线性的。原实现只取对数做单指数回归，慢成分整个丢掉。两个时间常数拉不开（比值约 2.7）时也不能退化成一个指数，否则 tau2/tau1 会掉到 1.8 以下。

对应测试：test_tau_pair_matches_independent_fit、test_near_degenerate_components_resolved

## 难点 3：限幅点要整点剔除

推理轴：数据口径

落在推理路径上的哪里：探测器上限以内的读数才代表真实波形，顶到上限的点是被限幅的。这些点如果留在拟合里，会把快成分的幅度与时间常数都拉偏。原实现全算进去了。

对应测试：test_saturated_points_removed、test_second_curve_with_other_rates

## 难点 4：基线、比值与残差的定义

推理轴：业务规则

落在推理路径上的哪里：模型里有一个与时间无关的本底 c，它和慢成分的长尾互相牵扯；报告里的幅度比要等于 a1/a2；残差按自由度 n-6 换算成 rmse，而且残差要放回实测读数上算（反解回来的空间里噪声被放大，不能拿来当 rmse）。原实现既没有基线，也没有这几栏。

对应测试：test_baseline_and_ratio_consistent、test_rmse_near_noise_level、test_report_schema_and_rounding

## 难点 6：标定点与瘫痪型堆积模型

推理轴：领域公式（两种堆积模型）

落在推理路径上的哪里：读出电路的堆积有两种模型，记录里的 `instrument.pileup_model` 写 `paralyzable` 时要换成 y = u·exp(-k·u)，这条没有闭式反解，得用数值迭代把 u 解出来；写成非瘫痪型那条式子，高计数段的形状整个对不上。记录若带 `calibration`（若干真实计数率 → 读数标定点），堆积系数必须先在标定点上单独定出来再反解曲线，把标定点丢掉、直接连曲线一起联合拟合，k 会被拟合残差带偏。

对应测试：test_calibration_pins_pileup_coefficient、test_paralyzable_pileup_model

## 跨运行状态（重复运行与不污染）

同一份记录重复运行，报告逐字节一致（不能写时间戳）；运行失败时不能覆盖上一次的报告：先写临时文件再原子替换，非法记录以退出码 2 收场且不落文件。对应测试：test_rerun_produces_identical_bytes、test_failed_run_keeps_previous_report。

非法输入另有 test_invalid_data_rejected_without_output 单独覆盖（点太少、时间不单调、上限非正、剔除后有效点不足）。

## 难点 5：跨运行的复核缓存

推理轴：跨运行状态

落在推理路径上的哪里：同一条曲线会被反复复核，结果要按记录字节的 sha256 缓存起来：命中就直接复用（不再重算），未命中就追加到末尾，其他条目原样保留；缓存读不出来或结构不对要当空缓存并从零重写；记录不合法时缓存一个字节都不许动；缓存与报告都先写临时文件再原子替换。原实现没有缓存，也不管原子写。

对应测试：test_cache_reuse_and_append、test_corrupt_cache_recovered、test_failed_run_leaves_cache_untouched

## 权重口径

17 条计分测试 + 1 条门禁（`test_invalid_data_rejected_without_output`，不通过则整份记 0），权重和 = 1。权重最高的三项是 标定点定堆积系数 0.125、时间常数对 0.083、瘫痪型堆积模型 0.073；逐条权重与理由写在 `tests/scoring.json` 里，其余各项按领域重要性递减分配：核心拟合与判定 > 边界与拒收 > 输出契约与重复运行。

## 定标依据（自检用）

判分侧独立做（tau1, tau2, k）粗网格加实测域 LM 精修（网格密度、初值与迭代路径都与参考解不同），同一份曲线上两者参数一致到 1% 内。本地实测参考解 1.000000、缺陷版 0.000000；缺陷版是单指数对数回归、不剔限幅点、不反解堆积、不做缓存、报告带时间戳且缺基线/比值/残差几栏，18 条全挂（17 条计分 + 1 条门禁）。
