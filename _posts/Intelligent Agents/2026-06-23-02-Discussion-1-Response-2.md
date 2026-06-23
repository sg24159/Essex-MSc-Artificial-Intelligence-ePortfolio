---
title: "02 Discussion 1: Agent Based Systems: Response 2"
category: Intelligent Agents
---

The push towards interconnected systems makes sense, moving from centralized systems to distributed decision-making reduces the latency between input and a decision, while also reducing task complexity by breaking problems down into simpler units (Marik and McFarlane, 2005).

84% is a big number, is it because organizations are discovering the benefits of distributed systems or is it excitement around LLM-based solutions? I am concerned that it is the latter given that the history of distributed agents is very long (Marik and McFarlane, 2005; An, 2026).

One of the disadvantages of distributed systems, is that they can be hard to simulate (Leitão, Mařík and Vrba, 2013) and produce unpredictable high level behaviour (Marik and McFarlane, 2005). Placing LLMs in this environment does not improve results, in fact LLMs themselves are hard to simulate and bring new failure modes, such as hallucinations and malformed tool calls (An, 2026). Rules based agents are easy to interpret and update with new behaviour simply by reviewing the rules (Sarker et al., 2024). Conversely, LLM-based agents require heavy logging to provide the same level of explainability, as well as layered protections against hallucinations, such as prompt engineering and Retrieval-Augmented Generation (RAG), introducing even more complexity into the system, which is difficult to maintain (Hiriyanna and Zhao, 2025).

It may be possible to get the best of both, by having a programmatic system layer coupled with an advisory LLM layer on top, this approach has seen success in cybersecurity (Sarker et al., 2024) and manufacturing (González-Potes et al., 2026).

**References**

An, B. (2026) “From Symbols to Synapses: The Reemergence of Agency in the Large Language Model Era,” IEEE Intelligent Systems, 41(2), pp. 31–38. Available at: https://doi.org/10.1109/MIS.2026.3668397.

González-Potes, A. et al. (2026) “Hybrid AI and LLM-Enabled Agent-Based Real-Time Decision Support Architecture for Industrial Batch Processes: A Clean-in-Place Case Study,” AI, 7(2), p. 51. Available at: https://doi.org/10.3390/ai7020051.

Hiriyanna, S. and Zhao, W. (2025) “Multi-Layered Framework for LLM Hallucination Mitigation in High-Stakes Applications: A Tutorial,” Computers, 14(8), p. 332. Available at: https://doi.org/10.3390/computers14080332.

Leitão, P., Mařík, V. and Vrba, P. (2013) “Past, Present, and Future of Industrial Agent Applications,” IEEE Transactions on Industrial Informatics, 9(4), pp. 2360–2372. Available at: https://doi.org/10.1109/TII.2012.2222034.

Marik, V. and McFarlane, D. (2005) “Industrial adoption of agent-based technologies,” IEEE Intelligent Systems, 20(1), pp. 27–35. Available at: https://doi.org/10.1109/MIS.2005.11.

Sarker, I.H. et al. (2024) “Multi-aspect rule-based AI: Methods, taxonomy, challenges and directions towards automation, intelligence and transparent cybersecurity modeling for critical infrastructures,” Internet of Things, 25, p. 101110. Available at: https://doi.org/10.1016/j.iot.2024.101110.

