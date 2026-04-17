# Mid-Weekly Recommendations

基于 `results/runs/20260416_170652/run_summary.csv`，当前真正成功训练过双 regime 模型的品种一共 25 个：

`AG AU B BB BU C CS CU EG FB FU JD JM M PB PG RB RR RU SN SP V WR Y ZN`

这份建议只覆盖这 25 个品种。

## 先说结论

- `mid_weekly` 在当前代码里不是自动推断的，而是逐品种从 `product_registry.json` 的 `mid_weekly_files` 显式读取。
- 同一个 `mid_weekly` 文件可以被多个品种复用，不需要每个品种都造一份重复文件。
- 因子会被 `merge_asof(..., direction="backward")` 后再 `ffill()` 到分钟级，所以更适合放慢频、供需、库存、开工、信用、景气度这类“慢变量”，不适合塞已经被分钟 K 线和成交量捕捉到的短周期技术信号。
- 当前 `data/mid_weekly/` 还是空目录，所以最合理的做法是：
  1. 先用仓库里已有的月频宏观表，导出一批可以马上落地的共享文件。
  2. 再按产业链逐步补每个品种最关键的周频库存/开工/基差。

## Mid-Weekly 文件约束

每个文件最好都满足下面的格式：

- 一列时间列，列名可为 `tdate/date/datetime/timestamp/trade_date`
- 一列数值列
- 文件名的 stem 会变成特征名，例如 `macro_pmi.csv` 会变成 `MID_macro_pmi`

所以命名尽量直接、稳定，不要把同一含义拆成很多版本。

## 第一层：现在就能落地的共享宏观文件

这些都可以直接从 `data/macro/macro_monthly_features_core.csv` 导出来，作为第一批 `mid_weekly` 文件。

| 建议文件名 | 来源列 | 为什么值得先做 |
| --- | --- | --- |
| `macro_ppi_yoy.csv` | `ths_PPI_当月同比` | 商品价格环境总开关，几乎对所有工业品都有帮助。 |
| `macro_cpi_yoy.csv` | `ths_CPI_当月同比` | 对贵金属、农产品和食品链更有用。 |
| `macro_pmi.csv` | `nbs_制造业采购经理指数_pct` | 宏观景气度确认，适合工业品。 |
| `macro_new_orders.csv` | `nbs_新订单指数_pct` | 比 PMI 总指数更贴近真实需求扩张。 |
| `macro_finished_goods_inventory_index.csv` | `nbs_产成品库存指数_pct` | 反映补库/去库阶段，对黑色、能化、有色都常有增量。 |
| `macro_industrial_production_yoy.csv` | `ths_规模以上工业增加值_当月同比` | 实体生产强度，适合有色、黑色、化工。 |
| `macro_fai_ytd_yoy.csv` | `nbs_固定资产投资额累计增长_pct` | 中游制造业和基建需求，适合金属和建材链。 |
| `macro_real_estate_starts_ytd_yoy.csv` | `nbs_房地产新开工施工面积累计增长_pct` | 对黑色、PVC、板材、沥青更关键。 |
| `macro_new_home_sales_ytd_yoy.csv` | `nbs_新建商品房销售面积累计增长_pct` | 地产后周期需求代理，对板材、PVC、橡胶也有参考价值。 |
| `macro_credit_impulse.csv` | `afre_社会融资规模存量_growth_pct` | 国内信用脉冲，常领先工业品需求和通胀预期。 |

## 第二层：逐品种建议

原则上每个品种先挂 2 到 3 个“能马上做”的共享宏观文件，再补 1 到 2 个真正有产业链辨识度的周频因子，效果通常比一口气堆很多慢变量更稳。

### 贵金属

| 品种 | 现在先加 | 后续优先补的周频 | 原因 |
| --- | --- | --- | --- |
| `AU` | `macro_cpi_yoy.csv`, `macro_ppi_yoy.csv`, `macro_credit_impulse.csv` | `gold_etf_holding.csv`, `us_real_yield_10y.csv` | 黄金更吃通胀预期和流动性环境；分钟级价格因子已经有了，慢变量要补“通胀/利率/资金”这条线。 |
| `AG` | `macro_cpi_yoy.csv`, `macro_ppi_yoy.csv`, `macro_credit_impulse.csv` | `silver_etf_holding.csv`, `gold_silver_ratio.csv` | 白银兼具贵金属和工业属性，既受通胀预期影响，也受风险偏好和金银比驱动。 |

### 有色

| 品种 | 现在先加 | 后续优先补的周频 | 原因 |
| --- | --- | --- | --- |
| `CU` | `macro_new_orders.csv`, `macro_industrial_production_yoy.csv`, `macro_fai_ytd_yoy.csv` | `shfe_cu_inventory.csv`, `copper_spot_premium.csv` | 铜是最典型的顺周期工业金属，订单、工业生产和投资强度都直接影响定价。 |
| `PB` | `macro_new_orders.csv`, `macro_industrial_production_yoy.csv`, `macro_fai_ytd_yoy.csv` | `shfe_pb_inventory.csv`, `lead_battery_output_yoy.csv` | 铅更贴近蓄电池和汽车链，宏观需求之外，库存和下游电池开工会明显补充解释力。 |
| `SN` | `macro_new_orders.csv`, `macro_industrial_production_yoy.csv`, `macro_credit_impulse.csv` | `shfe_sn_inventory.csv`, `electronics_production_yoy.csv` | 锡受电子制造链影响更强，信用和制造订单扩张时往往更敏感。 |
| `ZN` | `macro_new_orders.csv`, `macro_industrial_production_yoy.csv`, `macro_fai_ytd_yoy.csv` | `shfe_zn_inventory.csv`, `galvanized_sheet_output_yoy.csv` | 锌和镀锌、基建、制造强相关，库存与订单组合通常比单看价格更有前瞻性。 |

### 黑色和建材

| 品种 | 现在先加 | 后续优先补的周频 | 原因 |
| --- | --- | --- | --- |
| `RB` | `macro_real_estate_starts_ytd_yoy.csv`, `macro_fai_ytd_yoy.csv`, `macro_new_orders.csv` | `rebar_social_inventory.csv`, `steel_mill_inventory.csv` | 螺纹核心在地产和基建，库存数据又是最直接的产业链温度计。 |
| `WR` | `macro_real_estate_starts_ytd_yoy.csv`, `macro_fai_ytd_yoy.csv`, `macro_new_orders.csv` | `wire_rod_inventory.csv`, `steel_mill_inventory.csv` | 线材与螺纹的需求逻辑接近，先复用黑色地产链宏观，再补钢材周频库存。 |
| `JM` | `macro_new_orders.csv`, `macro_industrial_production_yoy.csv`, `macro_real_estate_starts_ytd_yoy.csv` | `coking_coal_port_inventory.csv`, `coke_inventory.csv` | 焦煤最终仍走向钢厂需求，订单和地产链景气能定方向，港口库存能定节奏。 |
| `BU` | `macro_real_estate_starts_ytd_yoy.csv`, `macro_fai_ytd_yoy.csv`, `macro_ppi_yoy.csv` | `asphalt_social_inventory.csv`, `asphalt_plant_operating_rate.csv` | 沥青兼有原油成本和基建需求双重属性，地产/基建慢变量加库存开工最合适。 |
| `BB` | `macro_real_estate_starts_ytd_yoy.csv`, `macro_new_home_sales_ytd_yoy.csv`, `macro_credit_impulse.csv` | `wood_panel_inventory.csv`, `home_furnishing_demand_index.csv` | 胶板的直接周频数据通常少，先挂地产后周期和信用代理变量最现实。 |
| `FB` | `macro_real_estate_starts_ytd_yoy.csv`, `macro_new_home_sales_ytd_yoy.csv`, `macro_credit_impulse.csv` | `fiberboard_inventory.csv`, `home_furnishing_demand_index.csv` | 纤板和地产竣工、家具链更相关，先用后周期需求代理，等有行业库存再补。 |

### 能化

| 品种 | 现在先加 | 后续优先补的周频 | 原因 |
| --- | --- | --- | --- |
| `EG` | `macro_new_orders.csv`, `macro_pmi.csv`, `macro_finished_goods_inventory_index.csv` | `eg_port_inventory.csv`, `polyester_operating_rate.csv` | 乙二醇本质上看聚酯链景气，订单和产成品库存能给出补库线索，港口库存和聚酯开工更直接。 |
| `PG` | `macro_new_orders.csv`, `macro_pmi.csv`, `macro_ppi_yoy.csv` | `lpg_port_inventory.csv`, `pdh_operating_rate.csv` | LPG 兼有燃料和化工原料属性，景气和成本环境先定大方向，港口库存和 PDH 开工补细节。 |
| `FU` | `macro_ppi_yoy.csv`, `macro_pmi.csv`, `macro_credit_impulse.csv` | `fuel_oil_inventory.csv`, `refinery_throughput.csv` | 燃油受成本端和炼厂开工影响大，单靠分钟价量往往抓不到基本面切换。 |
| `RU` | `macro_new_orders.csv`, `macro_new_home_sales_ytd_yoy.csv`, `macro_pmi.csv` | `ru_qingdao_inventory.csv`, `tire_operating_rate.csv` | 橡胶最终看轮胎和汽车链，青岛库存和轮胎开工是很强的慢变量。 |
| `V` | `macro_real_estate_starts_ytd_yoy.csv`, `macro_new_home_sales_ytd_yoy.csv`, `macro_new_orders.csv` | `pvc_social_inventory.csv`, `pvc_downstream_operating_rate.csv` | PVC 对地产需求敏感，但交易节奏往往由社会库存和下游开工决定。 |
| `SP` | `macro_new_orders.csv`, `macro_pmi.csv`, `macro_credit_impulse.csv` | `pulp_port_inventory.csv`, `paper_mill_operating_rate.csv` | 纸浆更接近制造业和包装纸链，订单与信用能定大背景，港口库存和纸厂开工补供需变化。 |

### 农产品和养殖

| 品种 | 现在先加 | 后续优先补的周频 | 原因 |
| --- | --- | --- | --- |
| `B` | `macro_cpi_yoy.csv`, `macro_credit_impulse.csv` | `soybean_port_inventory.csv`, `soybean_arrival_forecast.csv`, `soybean_crush_margin.csv` | 豆二真正有效的是进口到港、压榨和港口库存；宏观只建议放一个轻量代理，不要喧宾夺主。 |
| `M` | `macro_cpi_yoy.csv`, `macro_credit_impulse.csv` | `soymeal_inventory.csv`, `oil_mill_operating_rate.csv`, `hog_profit.csv` | 豆粕更受压榨节奏和养殖利润影响，慢变量应该围绕库存、开工、下游利润。 |
| `Y` | `macro_cpi_yoy.csv`, `macro_ppi_yoy.csv` | `soybean_oil_inventory.csv`, `palm_oil_port_inventory.csv`, `crush_margin.csv` | 豆油价格更多受油脂库存和压榨利润驱动，通胀环境只适合作为辅助。 |
| `C` | `macro_cpi_yoy.csv`, `macro_industrial_production_yoy.csv` | `corn_port_inventory.csv`, `deep_processing_profit.csv`, `feed_demand_index.csv` | 玉米既有饲料需求也有深加工需求，库存和加工利润比宏观总量更关键。 |
| `CS` | `macro_cpi_yoy.csv`, `macro_industrial_production_yoy.csv` | `corn_starch_inventory.csv`, `corn_starch_processing_profit.csv` | 玉米淀粉和下游加工利润、成品库存强相关，适合补一组加工链因子。 |
| `RR` | `macro_cpi_yoy.csv` | `rice_inventory.csv`, `rice_auction_volume.csv` | 粳米更偏政策和库存驱动，宏观景气帮助有限，优先找库存和拍卖投放数据。 |
| `JD` | `macro_cpi_yoy.csv` | `egg_inventory_days.csv`, `layer_hen_stock.csv`, `feed_cost_index.csv` | 鸡蛋最怕只看盘口；存栏、库存天数、饲料成本才是慢变量核心。 |

## 我建议的落地顺序

1. 先从 `macro_monthly_features_core.csv` 导出 8 到 10 个共享宏观文件，放进 `data/mid_weekly/`。
2. 优先给工业品里已经跑得最稳的品种挂上：
   - `RB`, `CU`, `RU`, `EG`, `V`, `JM`
3. 再给农产品里供需链最清楚的品种挂上：
   - `M`, `Y`, `C`, `JD`
4. `BB`, `FB`, `RR` 这类行业周频公开数据不太好找，先只挂共享宏观文件，不建议一开始硬塞很多质量不高的“伪周频”数据。

## 不建议的做法

- 不要把同一逻辑拆成很多高度重复的慢变量一起挂进去，例如同时放 5 个几乎等价的地产指标。
- 不要给所有品种统一挂完全相同的一组 `mid_weekly` 文件。
- 不要把已经在分钟级 runtime factors 里充分体现的价量型技术指标，再降采样一遍塞进 `mid_weekly`。
- 不要一开始就给每个品种挂 6 到 10 个慢变量；目前这套模型更适合每个品种先挂 2 到 4 个高辨识度因子。

## 最小可执行版本

如果你想先快速做一版，不想一次性补太多外部数据，我建议先只做下面这 10 个共享文件：

- `macro_ppi_yoy.csv`
- `macro_cpi_yoy.csv`
- `macro_pmi.csv`
- `macro_new_orders.csv`
- `macro_finished_goods_inventory_index.csv`
- `macro_industrial_production_yoy.csv`
- `macro_fai_ytd_yoy.csv`
- `macro_real_estate_starts_ytd_yoy.csv`
- `macro_new_home_sales_ytd_yoy.csv`
- `macro_credit_impulse.csv`

然后按上面的分品种建议，把这些共享文件先挂进 `product_registry.json`。等第一轮验证完，再逐步补每个产业链真正有辨识度的周频库存/开工/基差因子。
