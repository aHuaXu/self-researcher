1. 你怎么定义 Agentic Search 的 action space？
搜索、改写 query、阅读、引用、反思、停止回答，哪些是显式 action？
	
2. 搜索触发策略怎么训练？
是 supervised imitation、RL、DPO/IPO，还是 rule + learned router？
	
3. final answer reward 很稀疏时，你怎么做 credit assignment？
搜索动作的好坏如何从最终结果反推？
	
4. query generation 怎么评估？
query 本身没有唯一标准答案，你怎么判断一个 query 是好 query？
	
5. grounding reward 怎么避免 reward hacking？
模型引用大量无关来源但答案正确，怎么扣分？
	
6. 多轮搜索怎么控制停止条件？
怎么避免模型无限搜索，或者过早停止？

my answer：
- 设置最大的max_turns
- 过早停止/不搜索直接回答 -> 不搜索工具固定-0.3
	
7. 检索结果和模型参数知识冲突时，训练目标是什么？
永远相信检索？还是根据 source reliability 判断？

my answer：
- todo：要求输出
	
8. 你们的 verifier 怎么做？
LLM judge、NLI、规则、人工标注、还是 learned reward model？各自误差怎么处理？
	
9. 训练数据怎么构造？
真实用户 query、synthetic multi-hop、counterfactual outdated facts、conflict documents，各占多少？
	
10. 线上怎么做 routing？
哪些 query 走 Agentic Search，哪些走普通 RAG，哪些不搜？
	
11. latency 和 cost 怎么优化？
并行搜索、early stopping、cache、query rewrite reuse、retrieval budget 怎么设计？
	
12. 失败 case 最大的一类是什么？
是搜不到、搜错、读错、引用错、推理错，还是拒答策略错？
	
13. 你怎么证明模型真的因为 Agentic Search 变强，而不是因为用了更强的 base model 或更好的检索器？
	
14. 如果把搜索引擎换掉，policy 会不会崩？
	
你们的 search policy 对 retrieval backend 是否过拟合？
	
15. 如果用户问题本身有错误前提，系统怎么处理？
是顺着搜，还是先识别 premise error？
	
从面试官的维度，才真正能区分三种人：
	
第一种，只看过 Agentic Search 概念，会停在“模型自己决定什么时候搜”。
	
第二种，做过 RAG 工程，会讲检索、rerank、citation、latency。
	
第三种，真做过 Agentic Search/RL for tool use，会自然谈到 action space、credit assignment、reward hacking、verifier noise、routing、线上成本。