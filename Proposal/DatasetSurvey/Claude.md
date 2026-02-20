# Datasets for defending RAG systems against indirect prompt injection

**The most actionable finding: three open-source projects—SecAlign, StruQ, and BIPIA—together provide a complete, battle-tested pipeline** for generating synthetic training triplets, finetuning open-source LLMs with QLoRA, and evaluating Attack Success Rate against indirect prompt injection. Meta's SecAlign alone reduced injection ASR to below 10% on Llama-3-8B using only the publicly available Cleaned Alpaca dataset as a base, and its full training recipe is open-sourced. Combined with over 30 publicly available datasets catalogued below—spanning prompt injection attacks, RAG poisoning, instruction hierarchy evaluation, and system prompt security—researchers have more than enough resources to bootstrap 2,000–5,000 training triplets and rigorously evaluate defenses. This report organizes every relevant dataset by its role in the three project phases: seed data generation, finetuning, and evaluation.

---

## Foundational datasets for instruction hierarchy finetuning

Three datasets form the backbone of any instruction hierarchy defense project. Each has been validated in peer-reviewed work and directly supports the (System Instruction, User Query, Retrieved Data) triplet format.

**BIPIA (Benchmarking Indirect Prompt Injection Attacks)** from Microsoft Research is the single most relevant benchmark. Published at KDD 2025 (arXiv:2312.14197), it provides **626,250 training prompts and 86,250 test prompts** across five RAG-like tasks: Email QA, Web QA, Table QA, Summarization, and Code QA. Each sample pairs legitimate external content with embedded malicious instructions at varying positions. The repository at github.com/microsoft/BIPIA includes white-box adversarial finetuning code for Vicuna-7B/13B, making it directly adaptable for QLoRA. External content sources draw from OpenAI Evals (email), WikiTableQuestions, and Stack Exchange, giving realistic document distributions.

**SecAlign and Meta SecAlign** from Facebook/Meta Research (arXiv:2410.05451, CCS 2025) provide the most production-ready training methodology. SecAlign constructs preference triplets from the **Cleaned Alpaca dataset (~52K instruction-response pairs)** by injecting adversarial instructions into the data portion of each sample, then generating (secure output, insecure output) pairs for DPO training. The recipe uses 90% straightforward injection + 10% completion attacks. Meta SecAlign (July 2025, github.com/facebookresearch/Meta_SecAlign) extends this with a new `input` role for untrusted data, randomized injection positioning, and **LoRA-based finetuning** of Llama-3.1-8B-Instruct—directly comparable to the project's QLoRA approach. Open weights for Meta-SecAlign-8B and Meta-SecAlign-70B are released under Llama 3 community license.

**StruQ (Structured Queries)** from USENIX Security 2025 (arXiv:2402.06363, github.com/Sizhe-Chen/StruQ) takes a complementary SFT approach. It generates **52K training samples** from Cleaned Alpaca: 26K clean + 26K with simulated prompt injections using special delimiter tokens ([MARK], [INST], [COLN]) to separate instructions from data. Tested on Llama-7B and Mistral-7B-v0.1, its evaluation covers Naive, Ignore, Completion, and GCG attack types.

| Dataset | Size | Source | Training Method | Open Code |
|---------|------|--------|----------------|-----------|
| BIPIA | 712K prompts | github.com/microsoft/BIPIA | Adversarial SFT | ✅ |
| SecAlign | ~52K preference pairs | github.com/facebookresearch/SecAlign | DPO | ✅ |
| Meta SecAlign | ~52K + LoRA recipe | github.com/facebookresearch/Meta_SecAlign | DPO + LoRA | ✅ |
| StruQ | 52K structured samples | github.com/Sizhe-Chen/StruQ | SFT with delimiters | ✅ |
| Cleaned Alpaca | ~52K instruction pairs | github.com/gururise/AlpacaDataCleaned | Base dataset | ✅ |

---

## RAG poisoning and adversarial retrieval datasets

For generating realistic poisoned retrieved-context examples, several datasets simulate the exact threat model where malicious content enters through the retrieval pipeline.

**PoisonedRAG** (USENIX Security 2025, arXiv:2402.07867) is the most directly applicable RAG attack dataset. It operates on three standard knowledge bases—**Natural Questions (2.6M docs), HotpotQA (5.2M docs), and MS-MARCO (8.8M docs)**—generating 5 poisoned texts per target question across 100 questions per dataset. Both black-box and white-box (HotFlip-based) attacks achieve **>90% ASR** by injecting just 5 poisoned passages into million-scale corpora. The (target_question, poisoned_passage, attacker_answer) triplets can be directly repurposed as negative training examples for instruction hierarchy—teaching models that system instructions ("answer factually") override misleading retrieved content. Code at github.com/sleeepeer/PoisonedRAG auto-downloads BEIR-format datasets.

**Fujitsu's Agentic RAG Red Team Benchmark** (huggingface.co/datasets/Fujitsu/agentic-rag-redteam-bench, CC-BY-4.0) is a newer comprehensive resource with **~28K+ records** across four attack types. The B1 text-poison subset is most relevant: it includes `user_query`, `adversarial_goal`, `attack_subtype` (instruction smuggling, format tricks, encoding tricks, prioritization directives), `poison_payload`, and judge assessments for both baseline RAG and multi-turn agent configurations. This dataset provides ready-made evaluation data with automated judge annotations.

**Microsoft's LLMail-Inject Challenge** (huggingface.co/datasets/microsoft/llmail-inject-challenge, MIT license) offers **~462K submissions** from a SaTML 2025 competition where attackers crafted emails containing hidden instructions to cause an LLM email assistant to exfiltrate data. With 40 difficulty levels incorporating different defense configurations (SpotLighting, Prompt Shield, activation-based detection), this is the most realistic indirect prompt injection dataset available. Each record includes the email body (injection vector), structured objectives, model output, and scenario metadata.

**TrustRAG** (arXiv:2501.00879, github.com/HuichiZhou/TrustRAG) provides a defense framework that generates (query, clean_docs, poisoned_docs, correct_answer) tuples using K-means clustering for malicious document detection. While primarily a framework, its evaluation pipeline produces data directly usable for instruction hierarchy training across the same NQ/HotpotQA/MS-MARCO knowledge bases.

**HotFlip corpus poisoning** methods (EMNLP 2023, reproduced ECIR 2025) generate adversarial passages via gradient-based token perturbation. Even **10 adversarial passages fool >90% of queries** against Contriever. These passages represent a distinct attack class—semantically anomalous but highly retrievable—that complements the more natural-language injections from PoisonedRAG and BIPIA.

---

## Prompt injection attack and classification datasets

A rich ecosystem of prompt injection datasets exists on HuggingFace, ranging from small curated sets to massive competition archives. These serve primarily as seed data for synthetic triplet generation and as components of evaluation suites.

**SPML Chatbot Prompt Injection** (huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection, ~16K rows, MIT license) is uniquely structured with `System Prompt`, `User Prompt`, `Prompt injection` label, and a `Degree` field (0–10) indicating attack complexity. GPT-4 generates attacks by identifying system prompt rules, creating inverse rules, and generating subtle user prompts that violate them. This structure maps directly to the instruction hierarchy training format and enables curriculum-style training from simple to sophisticated attacks.

**Tensor Trust** (ICLR 2024, tensortrust.ai) contains **563K+ human-generated prompt injection attacks and 118K+ defenses** from an online game—the largest human-generated adversarial dataset for instruction-following LLMs. Columns include `pre_prompt`, `attack`, `post_prompt`, `llm_output`, and `is_prompt_extraction`. While focused on direct injection (prompt extraction and hijacking), the attack patterns generalize well as seed material.

**HackAPrompt** (EMNLP 2023, huggingface.co/datasets/hackaprompt/hackaprompt-dataset, MIT license) provides **600K+ adversarial prompts** from 2,800+ competition participants across multiple difficulty levels, tested against GPT-3.5, FlanT5-XXL, and text-davinci-003. Includes correctness labels and 29 documented prompt hacking techniques. Note: HiddenLayer analysis found significant label noise at higher levels.

**SaTML 2024 LLM CTF** (huggingface.co/datasets/ethz-spylab/ctf-satml24, MIT license) provides **137,063 multi-turn adversarial chats** with 44 accepted defenses. Each chat includes defense details, secret string, model used (GPT-3.5-turbo or Llama-2-70B), full message history, and `was_successful_secret_extraction` labels. Only 4% of extractions succeeded, providing rich negative examples.

For binary classification, the landscape includes **deepset/prompt-injections** (662 samples, Apache-2.0), **xTRam1/safe-guard-prompt-injection** (~10.6K samples with categorical taxonomy), **geekyrakshit/prompt-injection-dataset** (534K aggregated samples), and **gabrielchua/system-prompt-leakage** (354K samples with system prompt protection labels). The **Lakera datasets**—gandalf_ignore_instructions, mosscap_prompt_injection (hundreds of thousands from DEF CON 31), and gandalf_summarization—are all MIT-licensed and provide real-world attack diversity.

---

## Evaluation benchmarks for instruction hierarchy and defense robustness

Measuring defense effectiveness requires benchmarks that specifically test whether models maintain instruction priority under adversarial pressure.

**IHEval** (NAACL 2025, huggingface.co/datasets/zhihz0535/IHEval, arXiv:2502.08745, CC-BY-NC-ND-4.0) is the **only benchmark explicitly designed to evaluate instruction hierarchy following**. It contains ~3,538 examples across nine tasks defining the hierarchy as system message > user message > conversation history > tool output. Four test scenarios—rule following, NLP task execution, safety defense, and tool use—each test both aligned (instructions agree) and conflict (lower-priority sources attempt to override) settings. The tool-output conflict scenarios directly simulate RAG-style injection.

**LLM-PIEval** (Amazon, NeurIPS 2024 AdvML Workshop, github.com/amazon-science/llm-pieval) evaluates indirect prompt injection via **150 distinct API specifications** in RAG contexts. Attack strings are injected into retrieved knowledge from SQuAD context fields, testing whether LLMs produce valid malicious API calls. White-box and black-box modes measure injection success under different threat models.

**InjecAgent** (ACL Findings 2024) provides **1,054 test cases** covering 17 user tools and 62 attacker tools, benchmarking indirect injection in tool-integrated LLM agents. Two attack categories—direct harm and data exfiltration—test whether adversarial instructions in external content can manipulate LLMs into unauthorized tool calls.

**Meta's CyberSecEval 3** (huggingface.co/datasets/facebook/cyberseceval3-visual-prompt-injection, MIT license) includes **1,000 test cases** with explicit `injection_type` labels (direct vs. indirect) and `risk_category` (logic-violating vs. security-violating), plus system prompts and judge questions. Though designed for multimodal evaluation, the text components are independently useful.

For general capability retention during finetuning, **IFEval** (huggingface.co/datasets/google/IFEval, 541 prompts, Apache-2.0) tests verifiable instruction following and is a core component of Open LLM Leaderboard. **AlpacaEval** and **MT-Bench** provide standard utility benchmarks to ensure defense training doesn't degrade general performance.

| Benchmark | Focus | Size | Best For |
|-----------|-------|------|----------|
| IHEval | Instruction hierarchy conflicts | 3.5K | Primary hierarchy evaluation |
| BIPIA (test split) | Indirect PI in RAG tasks | 86K | RAG injection ASR |
| LLM-PIEval | API injection via RAG | 150 APIs | API-level injection ASR |
| InjecAgent | Agent tool-call injection | 1K | Tool-use injection ASR |
| CyberSecEval 3 | Direct/indirect injection types | 1K | Injection type classification |
| Fujitsu RAG Bench | RAG text/image poisoning | 28K+ | RAG-specific red-teaming |
| LLMail-Inject | Realistic email injection | 462K | Adaptive attack robustness |
| IFEval | Instruction following capability | 541 | Capability retention |

---

## How these datasets map to project phases

**Phase 1: Synthetic data generation (2,000–5,000 triplets).** Start with **Cleaned Alpaca** as the base instruction dataset and apply the SecAlign recipe: inject adversarial instructions into the data portion to create (system instruction, user query + retrieved data with injection, secure response, insecure response) quadruplets. Augment with attack patterns from **BIPIA** (5 RAG-like task domains), **SPML** (degree-graded attacks against system prompts), and **PoisonedRAG** (realistic poisoned passages). Mine **Tensor Trust** and **HackAPrompt** for diverse attack phrasing. Use **WildChat** (1M real conversations) for realistic user query distributions. The SecAlign paper demonstrates that this synthetic approach, requiring zero human labeling, is sufficient to achieve strong defense.

**Phase 2: QLoRA finetuning of Llama-3-8B and Qwen2.5-7B.** The **Meta SecAlign** repository provides a complete, proven LoRA finetuning recipe for Llama-3.1-8B-Instruct using DPO on synthetic preference data—directly transferable to Llama-3-8B with QLoRA. **StruQ** provides an alternative SFT approach with structured delimiter tokens. BIPIA's white-box defense code offers adversarial finetuning templates. The literature consensus is that **DPO (SecAlign-style) outperforms SFT alone** in reducing ASR against unseen attacks, though combining both approaches (StruQ structure + SecAlign preference optimization) may yield the best results.

**Phase 3: Evaluation.** Use **IHEval** as the primary benchmark for instruction hierarchy compliance. Measure indirect PI ASR on **BIPIA's test split** (86K prompts across 5 tasks) and **LLM-PIEval** (150 API specs). Test RAG-specific robustness with **Fujitsu's Agentic RAG Red Team Benchmark** and **PoisonedRAG** configurations. Validate against adaptive attacks using **LLMail-Inject** challenge data. Monitor capability retention with **IFEval** and **AlpacaEval**. Use **Garak** (github.com/NVIDIA/garak) for automated vulnerability scanning across dozens of attack categories.

---

## Conclusion

The open-source ecosystem for RAG prompt injection defense is remarkably mature as of 2025. **The critical insight is that you don't need to build from scratch**: Meta SecAlign provides a complete, validated pipeline from data generation through LoRA finetuning to evaluation, achieving sub-10% ASR on Llama-3-8B against both optimization-free and optimization-based attacks. BIPIA and PoisonedRAG supply RAG-specific attack scenarios that SecAlign's generic instruction-tuning base lacks. IHEval fills the evaluation gap with the only benchmark explicitly measuring instruction hierarchy compliance across priority levels. The most effective strategy combines SecAlign's DPO preference optimization with BIPIA's domain-diverse RAG attack scenarios, evaluated on IHEval's hierarchy-specific test cases—a combination no single prior work has explored, representing a genuine research contribution opportunity. All recommended datasets are publicly accessible, with permissive licenses (MIT, Apache-2.0, or CC-BY-4.0), and include working code repositories.