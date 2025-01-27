---
title: "09 Discussion 2: Reply 1: Generating and Recognizing Propaganda with Artificial Intelligence"
category: Machine Learning
---

The possibility of an AI system that can generate and spread targeted propaganda is a scary one.

LLMs do not necessarily require malicious training, they can be tricked into generating malicious content despite existing safety procedures. This can be accomplished by first generating the malicious text with benign keywords, and then performing text replacement. For example, and LLM might refuse to generate a sensational headline for “COVID Vaccine”, but will do as asked for “Tylenol” (Bianchi and Zou, 2024).

If a malicious actor does want to train a model to generate malicious content, they could take a shortcut and apply fine-tuning to an existing model. Research has shown that custom fine-tuning can bypass a model’s safety with as few as 10 examples, despite thousands or millions of data points used for safety tuning (Qi et al., 2023). It is also possible for benign fine-tuning to result in the model forgetting safety instructions, potentially allowing users to exploit a model that was created with good intentions (Qi et al., 2023).

Combating misinformation, both human and machine generated, is a task for which LLMs are well suited. LLMs have a significant amount of world knowledge, strong reasoning, and can be enhanced with external data sources. This makes them equipped for not only detecting misleading sentences, but also providing a detailed explanation of both the subject and the propaganda techniques employed (Chen and Shu, 2024; Daniel Gordon Jones, 2024).

**References**

Bianchi, F. and Zou, J. (2024) ‘Large Language Models are Vulnerable to Bait-and-Switch Attacks for Generating Harmful Content’. arXiv. Available at: https://doi.org/10.48550/arXiv.2402.13926.

Chen, C. and Shu, K. (2024) ‘Combating misinformation in the age of LLMs: Opportunities and challenges’, AI Magazine, 45(3), pp. 354–368. Available at: https://doi.org/10.1002/aaai.12188.

Daniel Gordon Jones (2024) ‘Detecting Propaganda in News Articles Using Large Language Models’, Engineering: Open Access, 2(1), pp. 01–12. Available at: https://doi.org/10.33140/EOA.01.02.10.

Qi, X., Zeng, Y., Xie, T., Chen, P.-Y., Jia, R., Mittal, P. and Henderson, P. (2023) ‘Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!’ arXiv. Available at: https://doi.org/10.48550/arXiv.2310.03693.