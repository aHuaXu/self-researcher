"""Generate sample research query parquet files for multi-agent training.

Produces training data compatible with verl/utils/dataset/rl_dataset.py's RLHFDataset.
Each row contains a research question as a chat-format prompt with metadata.

Output:
    data/multi-research.parquet       -- all queries
    data/multi-research_dev.parquet   -- first 5 queries (validation set)
"""

import os
import numpy as np
import pandas as pd

# Research questions across six domains, mixed Chinese and English
RESEARCH_QUERIES = [
    # --- tech ---
    {
        "domain": "tech",
        "content": "What are the key architectural differences between transformer-based and state-space models (e.g., Mamba) for long-context language modeling, and what are the trade-offs in terms of inference speed and quality?",
    },
    {
        "domain": "tech",
        "content": "大语言模型的推理优化技术有哪些？请比较量化、蒸馏、投机解码等方法的优缺点和适用场景。",
    },
    {
        "domain": "tech",
        "content": "How does retrieval-augmented generation (RAG) compare to long-context LLMs for knowledge-intensive tasks? What are the current best practices for production RAG systems?",
    },
    {
        "domain": "tech",
        "content": "多智能体系统（Multi-Agent Systems）在软件工程领域的最新应用进展如何？有哪些代表性框架和实际案例？",
    },
    # --- finance ---
    {
        "domain": "finance",
        "content": "What are the main approaches for using large language models in quantitative finance, including sentiment analysis, risk assessment, and automated trading strategy generation?",
    },
    {
        "domain": "finance",
        "content": "全球央行数字货币（CBDC）的发展现状如何？中国数字人民币与其他国家的CBDC项目有哪些异同？",
    },
    {
        "domain": "finance",
        "content": "How has the adoption of AI-driven credit scoring models affected financial inclusion and lending bias in emerging markets?",
    },
    # --- industry ---
    {
        "domain": "industry",
        "content": "What role does digital twin technology play in modern manufacturing, and what are the key challenges in scaling digital twins for complex supply chains?",
    },
    {
        "domain": "industry",
        "content": "工业互联网平台在智能制造中的应用现状和发展趋势是什么？请分析主要平台的技术架构差异。",
    },
    {
        "domain": "industry",
        "content": "How are autonomous mobile robots (AMRs) transforming warehouse logistics, and what are the latest advances in fleet coordination algorithms?",
    },
    # --- healthcare ---
    {
        "domain": "healthcare",
        "content": "What is the current state of AI-assisted drug discovery? How do foundation models for molecular property prediction compare to traditional virtual screening methods?",
    },
    {
        "domain": "healthcare",
        "content": "大型语言模型在医疗诊断辅助中的应用有哪些进展？如何解决医疗AI的可解释性和安全性问题？",
    },
    {
        "domain": "healthcare",
        "content": "How effective are federated learning approaches for training medical imaging models across hospitals while preserving patient privacy?",
    },
    {
        "domain": "healthcare",
        "content": "基因编辑技术CRISPR在遗传病治疗中的最新临床试验进展如何？主要的安全性挑战是什么？",
    },
    # --- science ---
    {
        "domain": "science",
        "content": "What are the recent breakthroughs in room-temperature superconductivity research, and how should we evaluate the reproducibility claims in this field?",
    },
    {
        "domain": "science",
        "content": "AI for Science的代表性成果有哪些？请分析AlphaFold、气象大模型等案例对科学研究范式的影响。",
    },
    {
        "domain": "science",
        "content": "How are large language models being used to accelerate materials discovery, and what datasets and benchmarks exist for evaluating their effectiveness?",
    },
    {
        "domain": "science",
        "content": "量子计算在组合优化和量子化学模拟方面的最新进展如何？目前的量子优势是否已经在实际问题中得到验证？",
    },
    # --- education ---
    {
        "domain": "education",
        "content": "How are adaptive learning systems powered by AI transforming personalized education, and what evidence exists for their effectiveness compared to traditional instruction?",
    },
    {
        "domain": "education",
        "content": "大语言模型在教育领域的应用前景和风险有哪些？如何在利用AI辅助教学的同时避免学术诚信问题？",
    },
    {
        "domain": "education",
        "content": "What are the most effective approaches for teaching programming to beginners using AI coding assistants, and how do they affect long-term learning outcomes?",
    },
    {
        "domain": "education",
        "content": "在线教育平台如何利用学习分析（Learning Analytics）技术提升学习效果？有哪些成功的实践案例和关键指标？",
    },
]


def build_dataframe(queries: list[dict]) -> pd.DataFrame:
    """Convert a list of query dicts into a DataFrame matching RLHFDataset format."""
    rows = []
    for i, q in enumerate(queries):
        prompt = np.array(
            [{"role": "user", "content": q["content"]}],
            dtype=object,
        )
        rows.append({
            "data_source": "research_query",
            "prompt": prompt,
            "ability": "research & analysis",
            "reward_model": {"ground_truth": "", "style": "research"},
            "extra_info": {
                "domain": q["domain"],
                "index": f"rq_{i:05d}",
            },
        })
    return pd.DataFrame(rows)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = build_dataframe(RESEARCH_QUERIES)

    train_path = os.path.join(data_dir, "multi-research.parquet")
    dev_path = os.path.join(data_dir, "multi-research_dev.parquet")

    df.to_parquet(train_path, index=False)
    print(f"Wrote {len(df)} queries to {train_path}")

    df_dev = df.iloc[:5].copy()
    df_dev.to_parquet(dev_path, index=False)
    print(f"Wrote {len(df_dev)} queries to {dev_path}")


if __name__ == "__main__":
    main()
