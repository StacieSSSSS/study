# fx_correlation position management — 2026-06-21

```
参数: entry_z=1.5, exit_z=0.5, momentum_lookback=5个交易日, momentum_threshold=0.3

EURUSD-USDKRW  [获利了结]  (conviction High 85.0)
    z=-0.06(neutral)  momentum=0.60(reverting)
    -> 价差已回归至均值附近——此前基于背离建立的仓位可以兑现
USDTWD-USDKRW  [获利了结]  (conviction High 73.3)
    z=-0.10(neutral)  momentum=0.95(reverting)
    -> 价差已回归至均值附近——此前基于背离建立的仓位可以兑现
AUDNZD-USDKRW  [获利了结]  (conviction Medium 65.0)
    z=-0.18(neutral)  momentum=0.68(reverting)
    -> 价差已回归至均值附近——此前基于背离建立的仓位可以兑现
EURUSD-USDTWD  [获利了结]  (conviction Medium 60.0)
    z=0.04(neutral)  momentum=0.97(reverting)
    -> 价差已回归至均值附近——此前基于背离建立的仓位可以兑现
GBPUSD-USDTWD  [买入]  (conviction Medium 53.3)
    z=0.55(moderate)  momentum=0.35(reverting)
    -> 价差偏离均值且处于极端区间但走势平稳，或处于中等区间且回归已确认
GBPUSD-USDKRW  [减仓]  (conviction Medium 50.0)
    z=0.24(neutral)  momentum=-0.89(extending)
    -> 价差已回到中性区间但又开始背离——没有边际优势，降低暴露
GBPUSD-USDJPY  [观望]  (conviction Medium 48.3)
    z=0.85(moderate)  momentum=-0.73(extending)
    -> 信号不够清晰（中性区间且无动量，或中等区间仍在背离）——暂不操作
USDCNY-USDTWD  [减仓]  (conviction Medium 46.7)
    z=-0.04(neutral)  momentum=-0.63(extending)
    -> 价差已回到中性区间但又开始背离——没有边际优势，降低暴露
EURUSD-GBPUSD  [谨慎加仓]  (conviction Medium 45.0)
    z=1.58(extreme)  momentum=-1.50(extending)
    -> 价差已是极端水平，但仍在继续背离——可能进一步走极端，控制加仓节奏
USDJPY-USDTWD  [获利了结]  (conviction Medium 45.0)
    z=0.02(neutral)  momentum=0.60(reverting)
    -> 价差已回归至均值附近——此前基于背离建立的仓位可以兑现
```
