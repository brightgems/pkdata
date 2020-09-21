# 中医药天池大数据竞赛——中医文献问题生成
数据集下载：
- http://cips-chip.org.cn/2020/eval5
- https://tianchi.aliyun.com/competition/entrance/531826/information

## 任务简介
随着自然语言处理技术的不断发展，问题自动生成(Question Generation)已经在很多实际应用场景中有落地（如在医药领域，可以应用到自动问诊、辅助诊疗等场景），通过机器主动提问可以用来高效构建或者补充知识库，扩大数据集规模。疫情催化下，人工智能正在持续助力中医药传承创新加速发展，在此背景下设置了面向中医药文本的“问题生成”挑战任务。

## 任务详情
本次标注数据全部来源于中医药领域文本，包括【黄帝内经翻译版】、【名医百科中医篇】、【中成药用药卷】、【慢性病养生保健科普知识】四个主要来源和部分中医论坛文本， 从5000篇文档中共标注了 13000对（问题、文档、答案）对，其中每篇文档由人工构造产生1～4对(问题, 答案)对。问题类型包括实体类和描述类两大类（是非类问题包含在描述类中），其中“问题“均由人工标注产生，“答案“均为文档中的连续片段。 本次任务由阿里云天池平台提供技术支持，其中3500篇语料及其标注数据将开放出来用做训练数据，750篇语料开放出来用于初赛阶段评估，剩余的750篇测试数据用于决赛阶段的评测，不再开放下载，进入决赛阶段的选手需按照阿里天池平台官方要求提交模型到天池运行环境进行在线测评。

## 数据说明
- 示范例子：
```
        {
        "id": 98,
        "text": "黄帝道：什麽叫重实？岐伯说：所谓重实，如大热病人，邪气甚热，而脉象又盛满，内外俱实，便叫重实",
        "annotations": [
        {
        "Q": "重实是指什么？",
        "A": "所谓重实，如大热病人，邪气甚热，而脉象又盛满，内外俱实，便叫重实"
        },
        {
        "Q": "重实之人的脉象是什么样？",
        "A": "脉象又盛满"
        }
        ],
        "source": "黄帝内经翻译版"
        },

        {
        "id": 714,
        "text":
        "葡萄胎：表现为剧烈恶心呕吐，阴道不规则流血，偶有水泡状胎块排出。子宫质软大多较停经月份大。妊娠合并急性胃肠炎：多有饮食不洁史，不仅有恶心呕吐，还常伴有腹痛、腹泻等胃肠道症状。孕痈：即妊娠期急性阑尾炎，表现为脐周或中上腹部疼痛，伴有恶心、呕吐，24小时内腹痛可转移到右下腹。",
        "annotations": [
        {
        "Q": "葡萄胎有什么症状表现？",
        "A": "表现为剧烈恶心呕吐，阴道不规则流血，偶有水泡状胎块排出。子宫质软大多较停经月份大。"
        },
        {
        "Q": "孕痈有什么表现？",
        "A": "表现为脐周或中上腹部疼痛，伴有恶心、呕吐，24小时内腹痛可转移到右下腹。"
        },
        {
        "Q": "孕痈有其它叫法吗？",
        "A": "妊娠期急性阑尾炎"
        }
        ],
        "source": "名医百科中医篇"
        },

        {
        "id": 1078,
        "text":
        "发热是小儿最常见的临床症状，是机体抵抗疾病的一种防御性反应，家长不必惊慌，也不要一发热就吃退热药，这样反而会影响疾病的诊断，也不利于炎症的控制，要掌握以下几点：\n（1）首先找一找原因。小婴儿在夏季是否衣服太多，包得太严，喂水太少等。也可以查一查耳朵内、脖子及全身皮肤（肛门周围）有无发红、发肿、疖子等。\n（2）可给予降温处理。解开衣服，让散热增加。用38℃的温水擦浴20分钟。也可用75%酒精加水一半（为30%酒精）擦腋下、腹股沟、颈部。也可用热水袋内加水和冰块，枕于头下，或冷水毛巾敷于额头部。\n（3）如体温不降，烦躁不安，应去医院治疗。\n（4）发热时，要多喂水，多喂易消化、清淡的食物，如咽喉红肿疼痛，喂凉饮食可减轻疼痛。体温下降后，要注意保暖和营养的摄入。",
        "annotations": [
        {
        "Q": "小儿发热能否必须吃退烧药？",
        "A": "发热是小儿最常见的临床症状，是机体抵抗疾病的一种防御性反应，家长不必惊慌，也不要一发热就吃退热药，这样反而会影响疾病的诊断，也不利于炎症的控制。"
        },
        {
        "Q": "小儿发热要掌握哪几点？",
        "A":
        "（1）首先找一找原因。小婴儿在夏季是否衣服太多，包得太严，喂水太少等。也可以查一查耳朵内、脖子及全身皮肤（肛门周围）有无发红、发肿、疖子等。（2）可给予降温处理。解开衣服，让散热增加。用38℃的温水擦浴20分钟。也可用75%酒精加水一半（为30%酒精）擦腋下、腹股沟、颈部。也可用热水袋内加水和冰块，枕于头下，或冷水毛巾敷于额头部。（3）如体温不降，烦躁不安，应去医院治疗。（4）发热时，要多喂水，多喂易消化、清淡的食物，如咽喉红肿疼痛，喂凉饮食可减轻疼痛。体温下降后，要注意保暖和营养的摄入。"
        },
        {
        "Q": "小儿发热什么情况下要去看医生？",
        "A": "体温不降，烦躁不安，应去医院治疗。"
        }
        ],
        "source": "慢性病养生保健科普知识"
        },
```

- 数据格式说明：以json格式提供，包括如下字段：
```
id: 段落id
text: 段落文本
annotations: 数组，包括多对（问题、答案）对
Q：问题
A：答案
```


## 评价标准：
本次评测以ROUGE-L 和BLUE-4，Rouge-L得分相同情况下，对比BLUE-4成绩。

## 报名方式及任务提交：
请在阿里天池平台进行报名：https://tianchi.aliyun.com/competition/entrance/531826/introduction

## 时间安排：
- 报名时间：8月25日—9月28日
- 训练数据&初赛数据发布时间：9月7日
- 初赛时间：9月7日-9月30日
- 复赛时间：10月10日-10月25日
- 评测论文提交时间：10月31日
- CHIP会议日期(评测报告及颁奖)：11月6日—8日

## 三方包

`rouge-l估计指标计算`
```
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```

## 外部语料
[黄帝内经翻译版](http://ewenyan.com/contents/more/hdnj.html)
[名医百科中医篇](https://www.baikemy.com/disease/list/0/0?diseaseContentType=I)
[中成药用药卷](https://www.sohu.com/a/164758048_464395)
[慢性病养生保健科普知识](http://m.yihu.com/index.php/act/daily_health/index.html?platformType=4&sourceType=0&sourceId=0#%E5%85%A8%E9%83%A8)

## 解决方案

- [baseline] seq2seq with attention
https://github.com/andrew-r96/BertQuestionGeneration
https://github.com/Maluuba/nlg-eval

- [Basic] pretrain and finetune with bart

- [Advance] pretrain and finetune with bart and graph embeding attention