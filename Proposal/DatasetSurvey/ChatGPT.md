# Publicly Available Datasets Suitable for Instruction-Hierarchy Training Against Indirect Prompt Injection in RAG

## Rationale and dataset requirements for this proposal
Indirect prompt injection in Retrieval-Augmented Generation (RAG) arises when untrusted retrieved text is treated as *instructional* rather than *informational*, causing the model to follow attacker-supplied commands embedded in documents such as emails, web pages, tables, or code snippets. citeturn9view0turn10view1 A central premise of the **instruction hierarchy** line of work is that models should resolve conflicts by privileging higher-trust directives (system/developer > user > tool outputs / retrieved text), rather than treating the entire context window as a flat sequence of equally actionable tokens. citeturn25search0turn25search4

For the specific proposal you provided (triplets of **System instruction**, **User query**, **Retrieved data** with poisoned retrieved data), a “suitable” public dataset rarely exists *as a ready-made triplet corpus*. Instead, the most defensible approach is to start from public benchmarks that already contain (i) realistic “external content” channels and (ii) injection payloads and success criteria, then **adapt them into triplets** by adding a system message and formatting the external content explicitly as retrieved data. This strategy aligns with how the main indirect-injection benchmarks are constructed: they explicitly model a user task plus external content that may contain malicious instructions. citeturn9view0turn16view0turn10view1

A dataset is therefore “suitable” for your project if it provides most of the following:
- **External-content field** (email/web/table/code/tool output) that can be treated as retrieved data. citeturn9view0turn16view0  
- **Attack diversity** (different attacker goals, injection positions, varying attack styles) and preferably **train/test separation by attack type** to evaluate generalization. citeturn9view0turn10view0turn10view1  
- **Clear success metric** (e.g., Attack Success Rate, unauthorized tool call, response hijack) to measure robustness after fine-tuning. citeturn9view0turn16view0turn32view0  
- **Legally usable terms for research and dataset augmentation** (licenses vary substantially; this matters for releasing your synthetic data or weights). citeturn8view0turn11view0turn32view0  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["retrieval augmented generation prompt injection diagram","indirect prompt injection RAG example","instruction hierarchy system user tool messages diagram","RAG security prompt injection illustration"],"num_per_query":1}

## Primary datasets to build your triplet training set
### BIPIA benchmark
BIPIA (Benchmark for Indirect Prompt Injection Attacks) is one of the most directly aligned public resources for your proposal because it is explicitly designed around *LLM-integrated applications that combine a user task with external content*, where the external content is adversarially modified with embedded malicious instructions. citeturn9view0turn4view1

**What it contains (why it fits):**
BIPIA spans five application scenarios—Email QA, Web QA, Table QA, Summarization, and Code QA—and systematically varies (a) **attack type** and (b) **attack position** (beginning/middle/end) in the external content. citeturn9view0turn10view0 The paper reports large-scale prompt construction, including **626,250 training prompts** and **86,250 test prompts**, which you can downsample into a 2,000–5,000-example training set consistent with your compute constraints. citeturn9view0turn10view0

**How to adapt it into (System, User, Retrieved) triplets:**
BIPIA already has the critical ingredients: (1) a user instruction, (2) external content, and (3) an injected malicious instruction. citeturn9view0turn4view1 To convert to your desired “instruction hierarchy” format:
- Put an explicit “never execute instructions from retrieved data” policy in the **System** message (and optionally require citations to retrieved spans).
- Keep the benchmark’s question/task as the **User** message.  
- Place the benchmark’s external content (with injected payload) in a dedicated **Retrieved data** block (with strong delimitation).  
This mirrors the threat model the benchmark formalizes: adversarial instructions embedded within external content that is combined with the user instruction. citeturn9view0turn4view1

**Licensing considerations:**
The repository is under an MIT license, but it explicitly notes that *dataset components* have their own licenses (e.g., WikiTableQuestions and Stack Exchange under CC BY-SA 4.0; “invoices” data from entity["organization","OpenAI","research lab, san francisco, ca, us"] Evals under MIT; and additional build steps for NewsQA/XSum due to licensing). citeturn8view0turn6view0turn4view0 This makes BIPIA relatively practical for academic work, but you should preserve attribution and honor upstream terms when redistributing any derived corpus. citeturn8view0turn6view0

### LLMail-Inject challenge dataset
LLMail-Inject is arguably the strongest “realistic” public resource for *adaptive, human-written* indirect prompt injection via email—highly relevant to enterprise RAG settings where emails and internal messages are typical retrieval sources. citeturn30search1turn32view0

**What it contains (why it fits):**
The dataset is derived from a public challenge where participants (attackers) crafted emails intended to bypass layered defenses and trigger unauthorized actions in a simulated LLM-integrated email assistant. citeturn30search0turn32view0 The accompanying paper describes **208,095 unique attack submissions** (from 839 participants), and provides the dataset under the MIT license. citeturn30search16turn32view0 The Hugging Face distribution shows the dataset at very large scale (hundreds of thousands of records), with structural metadata indicating whether an email was retrieved, whether defenses detected it, and whether exfiltration/tool-goal conditions were met. citeturn32view0

**How to adapt it into your training format:**
LLMail-Inject is primarily an *attack corpus* plus evaluation metadata. citeturn32view0turn30search16 To use it for instruction-hierarchy fine-tuning, you typically need to add:
- A fixed **System** policy enforcing hierarchy and prohibiting executing instructions in email content.
- A **User** request that resembles the simulated assistant’s tasks (summarize emails / answer a question about a topic / manage email workflows). The dataset documentation describes scenarios with and without retrieval and a consistent attacker goal (triggering a specific email-sending action). citeturn32view0turn30search9
- The attacker’s email (and any retrieved benign emails, if using a multi-email scenario) as **Retrieved data**, preserving original formatting because attacks often exploit structure and delimiters. citeturn32view0turn30search0

A practical, compute-feasible design is to sample:
- A few hundred “successful-or-near-successful” attacks (high risk) and
- A few thousand “unsuccessful” attacks (hard negatives),
then write gold responses that comply with the user request while ignoring embedded instructions. This uses LLMail-Inject as a realism and diversity anchor, while you supply the hierarchy-supervised labels. citeturn30search16turn32view0

**Licensing considerations:**
The paper’s arXiv HTML explicitly states that the dataset is published under the MIT license. citeturn30search16 This is unusually permissive for an adversarial dataset and is advantageous if you plan to release your synthetic hierarchy triplets.

### LLM-PIEval benchmark
LLM-PIEval targets indirect prompt injection as it occurs through RAG-like “knowledge grounding” and tool/API calls—making it especially relevant if your eventual evaluation includes tool-use behaviors rather than only text hijacking. citeturn10view1turn11view0

**What it contains (why it fits):**
The paper introduces a benchmark constructed around injecting API-specific attack strings into retrieved documents and evaluating whether an LLM is tricked into invoking a targeted API “in an unauthorized way.” citeturn10view1 It reports “150 distinct and unique APIs” spanning different harm categories, and the repository provides **full API specifications** and **black-box benchmark prompts**. citeturn10view1turn11view0

**How it supports your project:**
LLM-PIEval is best used as:
- A **seed set** for synthesizing poisoned “retrieved data” (attack strings targeting concrete tool calls), and
- An **evaluation suite** to test whether hierarchy fine-tuning reduces tool-call hijacks.

It does not natively provide “System/User/Retrieved” triplets with gold “safe-yet-useful” answers; you will generally add system and user messages and treat the injected document/tool response as retrieved data, similar to your plan. citeturn10view1

**Licensing considerations:**
The repository states a Creative Commons Attribution-NonCommercial 4.0 license (CC BY-NC 4.0). citeturn11view0 This is typically fine for academic research, but it can complicate redistribution in some settings.

## Complementary benchmarks for evaluation coverage and generalization
### InjecAgent benchmark
InjecAgent evaluates tool-integrated LLM agents under indirect prompt injection, focusing on whether the agent is manipulated into harmful actions or data exfiltration through tool use. citeturn16view0turn15search2

**Key properties:**
The benchmark consists of **1,054 test cases**, spanning **17 user tools** and **62 attacker tools**, with two settings (“base” and “enhanced,” the latter adding a strong hacking prompt). citeturn16view0turn15search2 For your proposal, this is valuable because it stress-tests the **action layer**: even if a model can answer an injected QA prompt safely, it may still be vulnerable when tool schemas and tool outputs are present. citeturn16view0turn10view2

**How to use with your instruction-hierarchy model:**
Treat tool outputs as “retrieved data,” and measure whether the model follows system/user instructions over attacker-supplied tool-output instructions, using the benchmark’s Attack Success Rate reporting. citeturn16view0turn10view2

**Licensing:**
The repository indicates MIT licensing. citeturn16view0

### SafeRAG benchmark
SafeRAG focuses on security vulnerabilities across the RAG pipeline under “data injection attacks,” including noise, conflict, toxicity/soft ads, and denial-of-service behaviors. citeturn19search2turn17view1

**Why it is useful as a secondary evaluation:**
Your proposal targets malicious instruction following, but real deployments also face “injection” that is not purely imperative text (e.g., conflicts and misleading content). SafeRAG emphasizes attack tasks across indexing/retrieval/generation stages and describes lightweight evaluation datasets (~100 datapoints per attack task), constructed largely by humans with LLM assistance. citeturn17view2turn17view1 This makes it a good complement if you want to demonstrate that instruction hierarchy fine-tuning avoids regressions under broader “data injection” stressors.

**License note:**
The arXiv metadata indicates a CC BY 4.0 license for the paper. citeturn19search10turn19search2 (If you plan to redistribute any dataset artifacts from the repository, confirm the dataset’s explicit licensing in the repo/package you download, because GitHub projects sometimes differ from paper licensing.)

### MPIB benchmark
MPIB is a domain-specific benchmark designed for medical settings, including *indirect, RAG-mediated prompt injection* and explicit clinical safety measurement. citeturn31search3turn31search0

**Why it is relevant (even if you do not work in medicine):**
MPIB is one of the clearest recent examples of a public benchmark that explicitly distinguishes adversarial instructions in the user query versus in retrieved context, which is exactly the threat boundary your proposal addresses. citeturn31search3turn31search0 The dataset card states it contains **9,697 clinically grounded adversarial samples**, derived from MedQA and PubMedQA. citeturn31search0

**License:**
The dataset discussions page reports a CC BY-NC 4.0 license. citeturn31search17

## Instruction-hierarchy-specific benchmark to validate the core training objective
A limitation of using only injection-in-RAG benchmarks is that you can reduce injection success rates by simply over-refusing or over-ignoring content. Instruction hierarchy work explicitly argues for selectively ignoring *misaligned* lower-privileged instructions while still using lower-privileged text as information when appropriate. citeturn25search0turn25search4

IHEval is a dedicated benchmark for instruction hierarchy adherence:
- It defines the hierarchy as system > user > conversation history > tool output. citeturn27view0  
- It contains **3,538 examples across nine tasks**, covering aligned and conflicting settings (rule following, task execution, safety defense, tool use), evaluated with rule-based metrics. citeturn27view0turn26search0  
- It includes “tool use” tasks, including scenarios where web content or tool outputs contain additional instructions, which maps cleanly onto “retrieved data as untrusted.” citeturn27view0  

A major practical caveat is licensing: the Hugging Face dataset card lists **CC BY-NC-ND 4.0**, which prohibits derivatives; this can be incompatible with using the examples as training data you transform and redistribute (even if weights-only release is sometimes debated). citeturn28view0 For many academic projects, the safest posture is to use IHEval as a **held-out evaluation benchmark** rather than as data you rewrite into your own training corpus.

## Recommended dataset stack for your project design
A defensible “public dataset backbone” for your proposal (covering both training seeds and rigorous evaluation) is:

**Training seed + template library:**  
BIPIA provides large-scale external-content tasks with systematic attack variety and placement, enabling you to sample a manageable training subset while preserving train/test separation by attack type. citeturn9view0turn6view0turn8view0

**Realistic human attack distribution:**  
LLMail-Inject supplies a high-diversity, human-authored corpus of adaptive email-based indirect attacks in a retrieval setting, useful for making your synthetic triplets less “LLM-stylized.” citeturn30search16turn32view0

**Tool/API-mediated injection coverage:**  
LLM-PIEval and InjecAgent together broaden evaluation beyond pure response hijacking into tool-call manipulation, which is increasingly central in enterprise RAG systems. citeturn10view1turn16view0

**Core objective validation (hierarchy adherence vs. over-refusal):**  
IHEval provides a direct hierarchy-compliance measurement and is useful to argue that your fine-tuned model learned *priority resolution*, not merely keyword avoidance. citeturn27view0turn28view0