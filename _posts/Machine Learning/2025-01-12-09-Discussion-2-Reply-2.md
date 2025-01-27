---
title: "09 Discussion 2: Reply 2: Addressing Bias in Artificial Intelligence"
category: Machine Learning
---

Bias can result from a number of sources, for example, the data labellers might not represent diverse perspectives and the data sets frequently come from internet resources, which over-represent young English speakers from developed countries (Zhou et al., 2024). This issue is also not new, vector word embeddings have been shown to represent sexist or racist relationships, and efforts have been made to correct them (Abramski et al., 2023).

A good example of LLMs inheriting bias from training data is the associations they form with the word “math”. The following words “boring”, “difficult”, “tedious”, “frustrating”, and “exasperating” were associated with “math” by GPT-3, demonstrating that it mirrors math anxiety found in society (Abramski et al., 2023). Later models such as GPT-3.5 and GPT-4 respond with fewer negative associations with “math”, demonstrating that these biases can be addressed (Abramski et al., 2023).

However, Kotek, Dockum and Sun (2023), found that GPT-4 exhibited more gender bias than GPT-3.5. The bias exhibited was also found to reflect societal perceptions better than reality, and amplified the bias beyond what exists in societal perceptions. They also found that LLMs do not admit to assumptions made when answering ambiguous questions until explicitly prompted, this may mislead users who did not recognize the ambiguity.

Reinforcement Learning with Human Feedback (RLHF) is a successful method for avoiding unwanted behaviour in LLMs, however, it is not able to fully correct biases in training data (Kotek, Dockum and Sun, 2023). Another approach for addressing bias is to encourage LLMs to explain their reasoning and provide them with access to reliable knowledge sources (Zhou et al., 2024). I believe that a combination of curating training data, RLHF, and training LLMs to rely on factual sources over associations found in the training data can address most bias issues.

**References**

Abramski, K., Citraro, S., Lombardi, L., Rossetti, G. and Stella, M. (2023) ‘Cognitive Network Science Reveals Bias in GPT-3, GPT-3.5 Turbo, and GPT-4 Mirroring Math Anxiety in High-School Students’, Big Data and Cognitive Computing, 7(3), p. 124. DOI https://doi.org/10.3390/bdcc7030124.

Kotek, H., Dockum, R. and Sun, D. (2023) ‘Gender bias and stereotypes in Large Language Models’, in Proceedings of The ACM Collective Intelligence Conference. CI ’23: Collective Intelligence Conference, Delft Netherlands: ACM, pp. 12–24. DOI: https://doi.org/10.1145/3582269.3615599.

Zhou, J., Müller, H., Holzinger, A. and Chen, F. (2024) ‘Ethical ChatGPT: Concerns, Challenges, and Commandments’, Electronics, 13(17), p. 3417. DOI: https://doi.org/10.3390/electronics13173417.