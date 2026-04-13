#!/usr/bin/env python3
"""Generate a structured error-analysis report for MuSiQue output shards."""

from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PATTERN_ORDER = [
    "multi_hop_intermediate_entity_error",
    "temporal_anchor_error",
    "answer_type_confusion",
    "geographic_hierarchy_relation_error",
    "alias_spelling_formatting_eval_issue",
    "wrong_answer_granularity",
    "overgenerated_noncanonical_answer",
]

PATTERN_META = {
    "multi_hop_intermediate_entity_error": {
        "title": "多跳链路选错中间实体",
        "description": "模型沿着看似合理但错误的中间实体继续推理，最后答案落在另一条链上。",
    },
    "temporal_anchor_error": {
        "title": "时间点或历史锚点错误",
        "description": "模型找到了相关实体，但把 abolished / first / current / last 这类时间锚点落错了。",
    },
    "answer_type_confusion": {
        "title": "答案类型混淆",
        "description": "题目要的是角色、机构、平台、原因等特定槽位，模型却回答了另一种类型。",
    },
    "geographic_hierarchy_relation_error": {
        "title": "地理层级或关系错误",
        "description": "模型把 city / county / province / country / seat / border relation 混成了同一层。",
    },
    "alias_spelling_formatting_eval_issue": {
        "title": "别名、拼写或格式导致的伪错误",
        "description": "答案语义基本正确，但因为别名、头衔、连字符或轻微格式差异未命中 golden。",
    },
    "wrong_answer_granularity": {
        "title": "回答粒度不对",
        "description": "答案过宽、过窄或落在相邻粒度，语义相关但不满足题目要求。",
    },
    "overgenerated_noncanonical_answer": {
        "title": "自由生成过度展开，未命中标准答案",
        "description": "回答包含正确片段，但加了额外说明或额外候选，导致严格匹配失败。",
    },
}

ERROR_CLASSIFICATIONS = {
    "f6564f948dc28ef7be445d610eb42c1a6a6042d47e62e7a6": {
        "pattern": "wrong_answer_granularity",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "把上位类别“Scottish clan”回答成若干实例成员，粒度偏下。",
    },
    "102eaee0943270c43f63b7d45a0088f6a8515e51a126d925": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "沿着 adult contemporary radio 这条链跳到了错误艺人。",
    },
    "5188d3c87c6fead7f796c62c3f1bd13ea8576bab724ea06d": {
        "pattern": "alias_spelling_formatting_eval_issue",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "“24:00 local time”与 gold“24:00”语义一致，属于格式扩写。",
    },
    "037dbe8ccac208d33dfbd28d6fdc319f88b1c2c2bd4ba060": {
        "pattern": "wrong_answer_granularity",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "给出了相近但不准确的区间，数值边界偏差明显。",
    },
    "936ced65be306aaccf844e11a0e0ef544d44a2a7edec8f14": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "未找到精确次数，只生成了模糊答案“multiple”。",
    },
    "70444bcfcd30c497b730b5b408990ea2c672ec03caa5235f": {
        "pattern": "temporal_anchor_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "历史事件计数落在了错误地点或错误时间片上。",
    },
    "e349bd0727c841c1b6a985bbf6c1b043efcad758e078e1df": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "把 Thelma and Louise 的演员关系链接错到了另一个配偶角色。",
    },
    "8daaf871ac6117a987a5461e5a0cfc553ef30e1f47800200": {
        "pattern": "alias_spelling_formatting_eval_issue",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "“adult Dick Grayson (Nightwing)”已包含 gold alias，本质是规范化问题。",
    },
    "f11250d4b392a30f6d22566d5167dee1464544f525c2975c": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "错误地把 wife 槽位落在了另一个角色名上。",
    },
    "6d57f1e891250f9822eaef55f6f86661ba65ea90141fdb3f": {
        "pattern": "geographic_hierarchy_relation_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "县级实体落错到了相邻 county，且同题跨运行从正确漂移为错误。",
    },
    "6e64c9c62d8e1170c6341cde12aa2fe75a35762012ab8d30": {
        "pattern": "geographic_hierarchy_relation_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "把出生地城市 Wenzhou 当成了题目要的县级答案。",
    },
    "d51c30bc9bcc9c9daace1edac5bc07ded7df39ad346f4c04": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "在应答阶段完全丢失了目标实体，最终输出 None。",
    },
    "de9c2fbc33bc6136a3a0598a7a1ad678882b4a8613f3900b": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "label 关系链落在了乐队旧厂牌而不是题目对应的目标厂牌。",
    },
    "865a174ebe32ad81f573feb122198c1f16e6f2fd6c3fdda5": {
        "pattern": "answer_type_confusion",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "题目要系统名称，模型给了功能描述，没落到 canonical name“FSMA”。",
    },
    "a39f4cbf6b6f7cb942cd2ce0e1234cae29855f2eaffc1be9": {
        "pattern": "geographic_hierarchy_relation_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "回答了 municipality，而题目要求的是更高层级的 territorial entity。",
    },
    "ebd4b995b91cadca20e7b463a7eeb03130f2aa06a4ac30a8": {
        "pattern": "geographic_hierarchy_relation_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "边界 county 关系判断错误，落到了相邻但不正确的县。",
    },
    "334f6b3c862e350efb163f95dd0e7f59e2928309909f66e7": {
        "pattern": "overgenerated_noncanonical_answer",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "回答里包含 gold 片段，但额外加入瑞典运营说明，严格匹配失败。",
    },
    "a6005b1ac2dec9c661b1dec2b0d4e27939a56dff3d049992": {
        "pattern": "alias_spelling_formatting_eval_issue",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "语义上等同于 gold，只是缺了前置介词“for”。",
    },
    "16de423a865eb739d3ec248429096218820feb176cb1b5af": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "抓到了相关历史事件，但不是题目问的那个“另外发生的事”。",
    },
    "79669636487de3810792f2a9a1b4a7c253da580013d1d2fb": {
        "pattern": "answer_type_confusion",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "题目要平台缩写，模型却把体育联盟名 NFL 当成答案。",
    },
    "665c0eff8918d3eeecbdd3439e19c078253bde37c2977f2c": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "多跳 nationality -> salary 链条最终查到了错误数值，并在日志中稳定复现。",
    },
    "ccbb33282bd009f7844ad8305fadb84e7ede51932fc6a36d": {
        "pattern": "answer_type_confusion",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "题目问“competition named after”，模型返回了人名 Billie Jean King 而非机构/赛事归属。",
    },
    "fa5051512fcaa4d026335b0dc206addf60f317d85cc8998b": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "同题跨运行给出了两个不同错误实体，说明链路定位本身不稳定。",
    },
    "e5d8a267444a8532a875af8cfb8b07e6f6d02d6e85148540": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "retriever 类型定位正确，但最终属性值取错，输出了错误寿命统计。",
    },
    "c48373896fd9da5b8e7f7565c18ffa921992f00d45147aeb": {
        "pattern": "answer_type_confusion",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "题目要 saints symbol，模型却返回了 manufacturer。",
    },
    "f9a50652b4e5d7000c42ae27b2477a54d2a88cfceec0ec3b": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "castle 所属城市链条走偏，只选了文档里唯一出现的城堡名。",
    },
    "f182c6355042dd18a7d42f9275c46f1ee0c985e5ae5c6001": {
        "pattern": "wrong_answer_granularity",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "给出现代机构名 College of Charleston，而 gold 要求其早期名称。",
    },
    "05f1867336770bf201cf9a88de60e2b9bdd893efe98d9593": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "band -> performed Pythons -> label 的链条落在了另一张专辑/时期。",
    },
    "e62b87ef31160a755763a2c665a516dbcbcf1f116050019b": {
        "pattern": "geographic_hierarchy_relation_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "county border relation 和 county seat 两步都容易走偏，最终落在 Oshkosh 而非 Green Bay。",
    },
    "7ac9fed94844e0922433d13032d684b7336219240499d6c1": {
        "pattern": "alias_spelling_formatting_eval_issue",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "“fleur de lis”与 gold 中的连字符形式属于同一别名。",
    },
    "8a986323dbd4e14655caf5775eb54c222bc8081f890cadb6": {
        "pattern": "geographic_hierarchy_relation_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "模型回答了国家 Tanzania，而题目需要相邻 province。",
    },
    "59f0697aa3d51324795d9f102b93756a0f7bc68f5ebf4922": {
        "pattern": "wrong_answer_granularity",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "给了“novels created by Robert Ludlum”这种上位描述，而不是具体 novel/title。",
    },
    "e3d8c2e853ea5a5ce78f9285d01c0b9b6ba832bd1027699d": {
        "pattern": "temporal_anchor_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "把学校开始军事训练的年份错写成后来的具体日期，并在日志中重复出现。",
    },
    "6c52099d688bcf9f3be62de4417112635370c9d72b748084": {
        "pattern": "temporal_anchor_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "独立参赛时间在多次运行间漂移，但都没有对准 2016。",
    },
    "0da908a928c375084638f1e0ab10c329679eaacae7e4ffe0": {
        "pattern": "temporal_anchor_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "把历史探险抵达时间完全跳成了现代年份 1986。",
    },
    "8267f95a2928cb63203d20da0079d98138d4f58281fc4e9e": {
        "pattern": "temporal_anchor_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "把规则制定机构的党派控制时间理解成了 1946 选举结果，而非 January 2015。",
    },
    "046a6c5856736e4dc6f532658038afabde83e9c71507832f": {
        "pattern": "wrong_answer_granularity",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "题目接受 decade-level gold“1970s”，模型给了具体年份 1977。",
    },
    "92a5f365ee7514ded1c6f5dac8470281083060c3e6ec7594": {
        "pattern": "temporal_anchor_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "国家链条和 abolition 时间同时偏离，最终落在 1941 而非 1989。",
    },
    "1c5e70656ffbd6ba33bd853ce406b5efd495e16a3610ed9f": {
        "pattern": "temporal_anchor_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "稳定把目标 kingdom 的废止时间回答成 1707，而非题目要求的 918。",
    },
    "296a82255560704f3fd8c8ea411827192d0eceddbf38ce4c": {
        "pattern": "temporal_anchor_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "last time 关系没有对准正确赛果，输出了较早年份 2005。",
    },
    "6b6d8f1f4a1c7ecab4f8c3f8205f8bfc531627d12c21a3c4": {
        "pattern": "temporal_anchor_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "同类足球时间题在 fallback 后直接退回到赛季年份，而非具体比赛日期。",
    },
    "c6fe007044674795d280bb474e2adc9dc8bacd8853a9cef5": {
        "pattern": "overgenerated_noncanonical_answer",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "回答里包含正确地点 Ko Phi Phi Le，但额外加入了第二个拍摄地。",
    },
    "096c9e28af3c91a52463c424203181b2a9c114c3b782dbf5": {
        "pattern": "geographic_hierarchy_relation_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "用具体城市 Ocala 回答了题目要的地区级答案“Northern Florida”。",
    },
    "19e28bddd1e5817780ebfba2ff23817578fb6e01289caa3a": {
        "pattern": "overgenerated_noncanonical_answer",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "长句里包含正确的 Atlantic Ocean，但 strict match 不接受描述性展开。",
    },
    "c858bf26a171d617a8decbb3a67e082b577a5a1166ca15b2": {
        "pattern": "geographic_hierarchy_relation_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "国家判断在相邻水体/地区链路上偏到了 Kenya。",
    },
    "b4a033f133679fb6a0a2351544933c5537546d947ad50c8f": {
        "pattern": "answer_type_confusion",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "几乎是整题理解错误，输出与题目槽位完全无关的财富新闻描述。",
    },
    "9f271f822294cc528b1f20904bb7694884af4084e086d20f": {
        "pattern": "answer_type_confusion",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "题目要角色，模型却回答了配音演员 Neil Patrick Harris。",
    },
    "a621901602313fe0bb3278547bfd2dc7a7eeb93905d04b78": {
        "pattern": "alias_spelling_formatting_eval_issue",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "多了 Rabbi 头衔，核心实体与 gold 一致。",
    },
    "dc7a928fdebf66b8a407ec96057d15ff1fc3a243bb3d87f1": {
        "pattern": "alias_spelling_formatting_eval_issue",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "多了定冠词 The，属于轻微规范化差异。",
    },
    "2b182c3ce4bb7bb2c28fc1f86d7a76591488e7d2aef1bb35": {
        "pattern": "answer_type_confusion",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "模型复述了 nationality，而不是题目要的作者姓名。",
    },
    "2df2586b40a9e5928817855c3b3a44c45b0809376372f38d": {
        "pattern": "temporal_anchor_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "三道 Timor-Leste 总统题都给了过时但高频的总统名字 José Ramos-Horta。",
    },
    "177dde027032c5c71bf88fd4676536c0bef7d79d43c32659": {
        "pattern": "temporal_anchor_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "同一国家/委员会模板题重复暴露了过时任期锚点。",
    },
    "648de9dea3ad81e8ac733acd312913385129ab16dd091834": {
        "pattern": "temporal_anchor_error",
        "likely_source": "reasoning",
        "eval_issue": False,
        "note": "同类模板再次把目标时点的总统答成旧任者。",
    },
    "9ad33ca0b5a3f7c880fc6f40db3d7de9c369111c302c9dfd": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "siblings 关系链没有走到目标演员，稳定输出错误 sibling。",
    },
    "f4f04befbe7bb6e1ca4055ffd495356951166e116a15b3de": {
        "pattern": "temporal_anchor_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "把历史语境中的“queen”错解成较新的王室状态，输出 Camilla。",
    },
    "db14f1320cc3442cb2416f65c437cd0ee74449baf687602b": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "spouse relation 走到了完全不相关的人物 Alan Yang。",
    },
    "0942aa56c94ac545c00849686583273efd664347084e9682": {
        "pattern": "alias_spelling_formatting_eval_issue",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "Abu Ubayda 与 gold Abu Ubaidah 属于常见转写差异。",
    },
    "efa29c58c4ed7403a39deb68f6f52c1fdcc617e0fbdd9a5c": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "adult contemporary radio 题稳定跳到了 Phil Collins，而不是 Michael Bublé。",
    },
    "78b24e393c76b4fcd88852e310548721c5eea12bb8848aca": {
        "pattern": "multi_hop_intermediate_entity_error",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "battle named after birth city 的历史链条选错，最终落到 Bernard Montgomery。",
    },
    "a4c716e09434218a16bdfed465322e369606bd38f23d815a": {
        "pattern": "overgenerated_noncanonical_answer",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "同时输出两位作者，其中包含 gold 可接受作者但又加了额外人名。",
    },
    "0531724f1691e4d2a1e819d1e68c14edb61a1036ee25a862": {
        "pattern": "overgenerated_noncanonical_answer",
        "likely_source": "evaluation",
        "eval_issue": True,
        "note": "返回了三位共同写作者，包含 gold 但没有收敛到唯一目标实体。",
    },
    "a3854b7b279e499bd9263a9c581a5efb025fea9efefa00e6": {
        "pattern": "answer_type_confusion",
        "likely_source": "fallback",
        "eval_issue": False,
        "note": "题目问原因，模型却返回地点 Venice。",
    },
}

PATTERN_REPRESENTATIVES = {
    "multi_hop_intermediate_entity_error": [
        "f11250d4b392a30f6d22566d5167dee1464544f525c2975c",
        "efa29c58c4ed7403a39deb68f6f52c1fdcc617e0fbdd9a5c",
        "fa5051512fcaa4d026335b0dc206addf60f317d85cc8998b",
    ],
    "temporal_anchor_error": [
        "1c5e70656ffbd6ba33bd853ce406b5efd495e16a3610ed9f",
        "6c52099d688bcf9f3be62de4417112635370c9d72b748084",
        "2df2586b40a9e5928817855c3b3a44c45b0809376372f38d",
    ],
    "answer_type_confusion": [
        "c48373896fd9da5b8e7f7565c18ffa921992f00d45147aeb",
        "9f271f822294cc528b1f20904bb7694884af4084e086d20f",
        "a3854b7b279e499bd9263a9c581a5efb025fea9efefa00e6",
    ],
    "geographic_hierarchy_relation_error": [
        "6e64c9c62d8e1170c6341cde12aa2fe75a35762012ab8d30",
        "e62b87ef31160a755763a2c665a516dbcbcf1f116050019b",
        "8a986323dbd4e14655caf5775eb54c222bc8081f890cadb6",
    ],
    "alias_spelling_formatting_eval_issue": [
        "5188d3c87c6fead7f796c62c3f1bd13ea8576bab724ea06d",
        "7ac9fed94844e0922433d13032d684b7336219240499d6c1",
        "0942aa56c94ac545c00849686583273efd664347084e9682",
    ],
    "wrong_answer_granularity": [
        "f6564f948dc28ef7be445d610eb42c1a6a6042d47e62e7a6",
        "59f0697aa3d51324795d9f102b93756a0f7bc68f5ebf4922",
        "096c9e28af3c91a52463c424203181b2a9c114c3b782dbf5",
    ],
    "overgenerated_noncanonical_answer": [
        "334f6b3c862e350efb163f95dd0e7f59e2928309909f66e7",
        "c6fe007044674795d280bb474e2adc9dc8bacd8853a9cef5",
        "0531724f1691e4d2a1e819d1e68c14edb61a1036ee25a862",
    ],
}

KEY_CASE_IDS = [
    "1c5e70656ffbd6ba33bd853ce406b5efd495e16a3610ed9f",
    "665c0eff8918d3eeecbdd3439e19c078253bde37c2977f2c",
    "efa29c58c4ed7403a39deb68f6f52c1fdcc617e0fbdd9a5c",
    "c48373896fd9da5b8e7f7565c18ffa921992f00d45147aeb",
    "fa5051512fcaa4d026335b0dc206addf60f317d85cc8998b",
    "6d57f1e891250f9822eaef55f6f86661ba65ea90141fdb3f",
    "2df2586b40a9e5928817855c3b3a44c45b0809376372f38d",
    "865a174ebe32ad81f573feb122198c1f16e6f2fd6c3fdda5",
]

KEY_CASE_NOTES = {
    "1c5e70656ffbd6ba33bd853ce406b5efd495e16a3610ed9f": "同一题在日志里两次都稳定输出 1707，说明模型把 Heptarchy 所属 kingdom 的历史终点系统性地锚到了错误年份。",
    "665c0eff8918d3eeecbdd3439e19c078253bde37c2977f2c": "salary 题两次都输出 37411，显示 nationality -> salary 这类数值查找在 fallback 后很容易稳定卡死在错误值。",
    "efa29c58c4ed7403a39deb68f6f52c1fdcc617e0fbdd9a5c": "adult contemporary radio 题跨运行稳定答成 Phil Collins，说明人物链路一旦选错，后续不会自我纠正。",
    "c48373896fd9da5b8e7f7565c18ffa921992f00d45147aeb": "这是典型槽位错位：题目问 symbol，模型一直答 manufacturer，说明 answer-type constraint 缺失。",
    "fa5051512fcaa4d026335b0dc206addf60f317d85cc8998b": "同一 qid 在日志里先后给出两个不同错误实体，反映出这类多跳科技实体题不仅会错，而且错得不稳定。",
    "6d57f1e891250f9822eaef55f6f86661ba65ea90141fdb3f": "这题在历史日志里曾答对 Kenton County，后一次却变成 Campbell County，说明当前配置下 even without fallback 也存在去稳定化现象。",
    "2df2586b40a9e5928817855c3b3a44c45b0809376372f38d": "三道 Timor-Leste 总统模板题都答成 José Ramos-Horta，像是被高频先验覆盖了题目对应时点。",
    "865a174ebe32ad81f573feb122198c1f16e6f2fd6c3fdda5": "模型知道 FDA food safety system 的概念描述，但没有收敛到名称 FSMA，说明生成端缺少 canonical-name 压缩。",
}


@dataclass
class Entry:
    qid: str
    question: str
    predicted_answer: str
    golden_answers: list[str]
    source_file: str
    source_index: int
    used_direct_fallback: str
    retrieval_call_count: str
    generation_call_count: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="output/musique/dense_chunk200_topk1_25_topk2_8",
        help="Directory containing the MuSiQue shard outputs to analyze.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["2.txt", "3.txt", "4.txt"],
        help="Shard files to include in the analysis, in chronological precedence order.",
    )
    parser.add_argument(
        "--stats-log",
        default="output/statistics_log.txt",
        help="Path to statistics_log.txt for run-history context.",
    )
    return parser.parse_args()


def parse_entries(path: Path) -> list[Entry]:
    blocks = [block.strip() for block in path.read_text().split("---") if block.strip()]
    entries: list[Entry] = []
    for source_index, block in enumerate(blocks):
        data = {}
        for line in block.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                data[key] = value
        entries.append(
            Entry(
                qid=data["qid"],
                question=data["question"],
                predicted_answer=data["predicted_answer"],
                golden_answers=ast.literal_eval(data["golden_answers"]),
                source_file=path.name,
                source_index=source_index,
                used_direct_fallback=data.get("used_direct_fallback", ""),
                retrieval_call_count=data.get("retrieval_call_count", ""),
                generation_call_count=data.get("generation_call_count", ""),
            )
        )
    return entries


def history_status(history: list[str]) -> str:
    statuses = []
    if any("|SUCCESS|" in line for line in history):
        statuses.append("SUCCESS")
    if any("|FAILURE|" in line for line in history):
        statuses.append("FAILURE")
    return "+".join(statuses) if statuses else "NONE"


def duplicate_run_status(entries: list[Entry]) -> str:
    if len(entries) == 1:
        return "single_run"
    if len({entry.predicted_answer for entry in entries}) == 1:
        return "duplicate_same_answer"
    return "duplicate_changed_answer"


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def format_history(history: Iterable[str]) -> str:
    return " || ".join(history)


def iter_all_entries(result_dir: Path, shard_files: list[str]) -> Iterable[Entry]:
    for shard_name in shard_files:
        yield from parse_entries(result_dir / shard_name)


def summarize_records(result_dir: Path, shard_files: list[str], stats_log: Path):
    log_lines = stats_log.read_text().splitlines()
    by_qid: dict[str, list[Entry]] = defaultdict(list)
    for shard_name in shard_files:
        shard_path = result_dir / shard_name
        for entry in parse_entries(shard_path):
            by_qid[entry.qid].append(entry)

    question_rows = []
    error_rows = []
    pattern_counter = Counter()
    summary = Counter()

    for qid, runs in sorted(by_qid.items(), key=lambda item: item[1][-1].question.lower()):
        final = runs[-1]
        history = [line for line in log_lines if final.question in line]
        strict_result = "correct" if final.predicted_answer in final.golden_answers else "error"
        earlier_outputs = " || ".join(
            f"{run.source_file}:{run.predicted_answer}" for run in runs[:-1]
        )
        classification = ERROR_CLASSIFICATIONS.get(qid) if strict_result == "error" else None
        if strict_result == "error" and classification is None:
            raise KeyError(f"Missing classification for error qid {qid}")

        row = {
            "qid": qid,
            "question": final.question,
            "final_predicted_answer": final.predicted_answer,
            "golden_answers": " | ".join(final.golden_answers),
            "strict_result": strict_result,
            "appears_in_files": " | ".join(run.source_file for run in runs),
            "duplicate_run_status": duplicate_run_status(runs),
            "earlier_outputs": earlier_outputs,
            "used_direct_fallback": final.used_direct_fallback,
            "retrieval_call_count": final.retrieval_call_count,
            "generation_call_count": final.generation_call_count,
            "history_count": str(len(history)),
            "history_statuses": history_status(history),
            "history_repeated_same_answer": str(
                len(history) > 1 and len({line.rsplit("|", 1)[-1] for line in history}) == 1
            ),
            "history_in_statistics_log": format_history(history),
            "pattern": classification["pattern"] if classification else "",
            "likely_source": classification["likely_source"] if classification else "",
            "eval_or_normalization_issue": str(classification["eval_issue"]) if classification else "",
            "analyst_note": classification["note"] if classification else "",
        }
        question_rows.append(row)

        if strict_result == "correct":
            summary["strict_correct"] += 1
            if any("|FAILURE|" in line for line in history):
                summary["strict_correct_with_failure_log"] += 1
            continue

        summary["strict_errors"] += 1
        pattern_counter[classification["pattern"]] += 1
        if classification["eval_issue"]:
            summary["eval_like_errors"] += 1
        if final.used_direct_fallback == "True":
            summary["strict_errors_with_fallback"] += 1
        if len(runs) > 1:
            summary["strict_errors_duplicate_qids"] += 1
        if len(history) > 1:
            summary["strict_errors_multiple_log_entries"] += 1
        if len(history) > 1 and len({line.rsplit("|", 1)[-1] for line in history}) == 1:
            summary["strict_errors_same_logged_answer_repeated"] += 1
        if len(runs) > 1 and len({run.predicted_answer for run in runs}) > 1:
            summary["strict_errors_changed_prediction_across_files"] += 1
        if any("|SUCCESS|" in line for line in history):
            summary["strict_errors_with_success_log"] += 1
        if any("|FAILURE|" in line for line in history):
            summary["strict_errors_with_failure_log"] += 1

        error_rows.append(row)

    summary["unique_qids"] = len(question_rows)
    summary["raw_outputs"] = sum(1 for _ in iter_all_entries(result_dir, shard_files))
    summary["duplicate_qids"] = sum(row["duplicate_run_status"] != "single_run" for row in question_rows)
    summary["duplicate_same_answer"] = sum(
        row["duplicate_run_status"] == "duplicate_same_answer" for row in question_rows
    )
    summary["duplicate_changed_answer"] = sum(
        row["duplicate_run_status"] == "duplicate_changed_answer" for row in question_rows
    )

    return question_rows, error_rows, summary, pattern_counter


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def representative_lines(error_lookup: dict[str, dict[str, str]], qids: list[str]) -> list[str]:
    lines = []
    for qid in qids:
        row = error_lookup[qid]
        repeat_note = ""
        if row["history_count"] != "0" and int(row["history_count"]) > 1:
            repeat_note = f"; history={row['history_statuses']} x{row['history_count']}"
        lines.append(
            f"- `{row['question']}`: `{row['final_predicted_answer']}` vs `{row['golden_answers']}`{repeat_note}. {row['analyst_note']}"
        )
    return lines


def build_report(
    error_rows: list[dict[str, str]],
    summary: Counter,
    pattern_counter: Counter,
    result_dir: Path,
    shard_files: list[str],
) -> str:
    error_lookup = {row["qid"]: row for row in error_rows}
    pattern_table = ["| Pattern | Count | Share of 62 errors |", "| --- | ---: | ---: |"]
    for pattern in PATTERN_ORDER:
        count = pattern_counter[pattern]
        pattern_table.append(
            f"| {PATTERN_META[pattern]['title']} | {count} | {percent(count, summary['strict_errors'])} |"
        )

    strict_errors_with_no_fallback = summary["strict_errors"] - summary["strict_errors_with_fallback"]
    lines = [
        "# MuSiQue 100题 Error Analysis",
        "",
        f"分析对象：`{result_dir}` 下的 `{', '.join(shard_files)}`。",
        "",
        "## 第一部分：数据总览",
        "",
        "- 当前分析基于 `116` 条原始输出去重后的 `100` 个唯一 `qid`。最终核对结果是 `62` 个严格错误、`38` 个严格正确。此前计划里的 `61/39` 来自更早一轮抽样，按这次最终去重核对后应更新为 `62/38`。",
        "- `16` 个 `qid` 在多个文件中重复出现，其中 `13` 个重复题保持同一答案，`3` 个重复题在不同文件间发生了答案漂移。",
        f"- `23/62` 个严格错误触发了 `direct_fallback`，另外 `39/62` 个严格错误在 `statistics_log.txt` 里曾被记成 `SUCCESS`，说明当前日志状态并不能可靠代表答案正确性。",
        f"- `14/62` 个严格错误在日志里出现过多次，`10` 个错误是同一个错误答案被重复输出。另有 `1` 道严格正确题在日志里曾被记为 `FAILURE`。",
        f"- 我把 `12/62` 个错误归为更像 `evaluation / normalization` 的伪错误，其余 `50` 个更像真实的 reasoning / retrieval / fallback 问题。",
        "",
        "## 第二部分：逐题总表",
        "",
        "- 全量 `100` 题总表写入 `error_analysis_question_summary.csv`。",
        "- 严格错误子表写入 `error_analysis_strict_errors.csv`，包含 pattern、likely_source、history 和分析备注。",
        "- 去重规则：同一 `qid` 若出现在多个文件中，按文件顺序 `2.txt -> 3.txt -> 4.txt` 的最后一次结果作为当前结果，Earlier outputs 保留在 CSV 中。",
        "",
        "## 第三部分：错误 Pattern 总结",
        "",
        *pattern_table,
        "",
        "### 1. 多跳链路选错中间实体",
        f"- 共有 `{pattern_counter['multi_hop_intermediate_entity_error']}` 题，是最大的错误簇。典型现象是前一跳实体看起来合理，但后续一路在错误分支上完成推理。",
        *representative_lines(error_lookup, PATTERN_REPRESENTATIVES["multi_hop_intermediate_entity_error"]),
        "",
        "### 2. 时间点或历史锚点错误",
        f"- 共有 `{pattern_counter['temporal_anchor_error']}` 题。模型经常找到相关人物/国家/比赛，但把 abolished / first / current / last 的时间锚点落错。",
        *representative_lines(error_lookup, PATTERN_REPRESENTATIVES["temporal_anchor_error"]),
        "",
        "### 3. 答案类型混淆",
        f"- 共有 `{pattern_counter['answer_type_confusion']}` 题。最常见的是题目要角色、平台、机构名或原因，模型却回答演员、联盟、公司或地点。",
        *representative_lines(error_lookup, PATTERN_REPRESENTATIVES["answer_type_confusion"]),
        "",
        "### 4. 地理层级或关系错误",
        f"- 共有 `{pattern_counter['geographic_hierarchy_relation_error']}` 题。county / city / province / country / seat / border 这些地理层级经常被混淆。",
        *representative_lines(error_lookup, PATTERN_REPRESENTATIVES["geographic_hierarchy_relation_error"]),
        "",
        "### 5. 别名、拼写或格式导致的伪错误",
        f"- 共有 `{pattern_counter['alias_spelling_formatting_eval_issue']}` 题。答案语义上基本正确，但因为头衔、连字符、拼写变体或额外括注未命中 gold。",
        *representative_lines(error_lookup, PATTERN_REPRESENTATIVES["alias_spelling_formatting_eval_issue"]),
        "",
        "### 6. 回答粒度不对",
        f"- 共有 `{pattern_counter['wrong_answer_granularity']}` 题。模型能定位到相关对象，但回答的是过宽、过窄或历史命名不一致的粒度。",
        *representative_lines(error_lookup, PATTERN_REPRESENTATIVES["wrong_answer_granularity"]),
        "",
        "### 7. 自由生成过度展开，未命中标准答案",
        f"- 共有 `{pattern_counter['overgenerated_noncanonical_answer']}` 题。回答里常含有 gold 片段，但因为额外解释或额外候选，严格匹配判错。",
        *representative_lines(error_lookup, PATTERN_REPRESENTATIVES["overgenerated_noncanonical_answer"]),
        "",
        "### 8. direct fallback 后稳定输出错误答案",
        f"- `23/62` 个严格错误触发了 fallback，说明 fallback 不是兜底提效，而是当前错误的重要来源。至少 `10` 个错误在历史日志里重复输出了完全相同的错误答案。",
        "- 代表性重复错误包括：`multiple`（plague 次数题）、`Roxanne`（Grown Ups 关系题）、`Wenzhou`（县级地理题）、`37411`（salary 题）、`Sazerac Company`（symbol 槽位错位）、`1707`（Heptarchy 时间题）、`Tammy O'Rourke`（sibling 题）和 `Phil Collins`（adult contemporary radio 题）。",
        f"- 对比来看，没有 fallback 的严格错误仍有 `{strict_errors_with_no_fallback}` 个，说明主问题不止是 fallback 本身，还包括 tree reasoning 选链、答案槽位约束和最终答案压缩。",
        "",
        "## 第四部分：重点案例",
        "",
    ]

    for qid in KEY_CASE_IDS:
        row = error_lookup[qid]
        lines.extend(
            [
                f"- `{row['question']}`",
                f"  预测：`{row['final_predicted_answer']}`；标准：`{row['golden_answers']}`。",
                f"  分析：{KEY_CASE_NOTES[qid]}",
            ]
        )

    lines.extend(
        [
            "",
            "## 第五部分：可行动结论",
            "",
            "- `reasoning` 是主问题而不是单纯 retrieval miss。很多错误都给出“看起来合理”的实体，但答案槽位、时间锚点或中间实体选错，导致 `39` 个严格错误甚至在日志里被标成了 `SUCCESS`。",
            "- `fallback` 需要更强的约束。现在 fallback 常直接产出一个单点猜测，而且在多次运行中会稳定复现同一个错误答案，说明它缺少证据充足度检查和 answer-type validation。",
            "- `evaluation / normalization` 仍值得修。至少 `12` 个错误更像评测口径问题，如果增加 alias、contains-gold、title stripping、hyphen normalization 和括注清洗，strict error 数可以明显下降。",
            "- 针对模型侧改进，最优先的是三类约束：`answer type validation`、`temporal consistency checks`、`canonical short answer compression`。这三类改进分别对应最大簇的类型混淆、时间锚点错位和过度展开答案。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result_dir = (repo_root / args.result_dir).resolve()
    stats_log = (repo_root / args.stats_log).resolve()

    question_rows, error_rows, summary, pattern_counter = summarize_records(
        result_dir=result_dir,
        shard_files=args.files,
        stats_log=stats_log,
    )

    summary_path = result_dir / "error_analysis_question_summary.csv"
    errors_path = result_dir / "error_analysis_strict_errors.csv"
    report_path = result_dir / "error_analysis_report.md"

    write_csv(summary_path, question_rows)
    write_csv(errors_path, error_rows)
    report_path.write_text(
        build_report(
            error_rows=error_rows,
            summary=summary,
            pattern_counter=pattern_counter,
            result_dir=result_dir,
            shard_files=args.files,
        )
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {errors_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
