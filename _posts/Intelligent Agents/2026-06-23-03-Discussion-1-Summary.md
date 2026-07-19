---
title: "03 Discussion 1: Agent Based Systems: Summary: Oversight is Key"
category: Intelligent Agents
---

Sarka, John-Jon and Vinod had a few common threads in their responses. First, some domains are more sensitive to agent non-determinism than others. Second, hybrid architectures may be effective in combatting non-determinism. Third, validation and oversight are key measures when using non-deterministic agents.

Healthcare is the most obvious domain where a fully autonomous agent could be dangerous, but it is by no means the only one. For example, a support chatbot could misrepresent a product from a financial services company, which would have consequences ranging from lost sales to reputational harm or lawsuits (Hiriyanna and Zhao, 2025). Care must also be taken to mitigate agent biases, which could lead to harms in employment opportunities and access to the aforementioned healthcare and financial services (Shi et al., 2024).

John-Jon went in an interesting direction, LLMs for communication, and formal structures for execution. This has a lot of potential in the manufacturing context, LLMs can synthesize answers across system logs and check if there are correlations with alerts without getting fatigued. An unassisted human operator often learns to ignore repeated alerts assuming transient issues, and does not check if the logs show signs of pending failure. By putting an LLM on top of existing rules-based programs, they can act as decision support agents, aiding human operators, without the risks associated with granting LLMs the ability to execute tasks (González-Potes et al., 2026).

Validation can take many forms, but a good first step is ensuring that the LLM can understand the risks involved. For example, using keywords associated with danger like “force” and “privilege escalation” instead of benign terms like “enable” in tool names makes a significant different in an agent’s willingness to undertake harmful actions, even if the warnings in the tool’s description remain the same (Sehwag et al., 2025). Oversight is the last line of defense when using an LLM for decision-making, the logging system should include all queries, responses, sources, and decision factors in order to provide the transparency required to monitor performance and bias (Shi et al., 2024; Hiriyanna and Zhao, 2025)


**References**

González-Potes, A. et al. (2026) “Hybrid AI and LLM-Enabled Agent-Based Real-Time Decision Support Architecture for Industrial Batch Processes: A Clean-in-Place Case Study,” AI, 7(2), p. 51. Available at: https://doi.org/10.3390/ai7020051.

Hiriyanna, S. and Zhao, W. (2025) “Multi-Layered Framework for LLM Hallucination Mitigation in High-Stakes Applications: A Tutorial,” Computers, 14(8), p. 332. Available at: https://doi.org/10.3390/computers14080332.

Sehwag, U.M. et al. (2025) “PropensityBench: Evaluating Latent Safety Risks in Large Language Models via an Agentic Approach.” arXiv. Available at: https://doi.org/10.48550/arXiv.2511.20703.

Shi, D. et al. (2024) “Large Language Model Safety: A Holistic Survey.” arXiv. Available at: https://doi.org/10.48550/arXiv.2412.17686.
