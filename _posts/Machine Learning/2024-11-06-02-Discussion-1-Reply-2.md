---
title: "02 Discussion 1: Reply 2: Using AI for Incident Response"
category: Machine Learning
---

The report by the Uptime Institute (2022) raises some concerning issues: Serious and severe outages are rising in both frequency and length. The issues frequently occur with external providers.

The traditional response process to an outage involves locating the root cause by manually tracing backwards from affected services (Wang et al., 2021). However, this task is becoming increasingly difficult due to the large amounts of interdependent services in modern cloud systems (Chen et al., 2019; Wang et al., 2021). This can lead to wasted time as teams responsible for a failing service will determine that the issue lies with another service, and pass the responsibility to another team. In 52% of outages, the service that first has an issue is not the root cause (Wang et al., 2021).

These providers should consider incorporating AI techniques into their incident response workflows. Some approaches include using Bayesian networks and gradient boosted trees for diagnosis and early warning (Chen et al., 2019), incident correlation graphs for root cause analysis (Wang et al., 2021), and GPT-based models to create a summary of the incident (Jin et al., 2023). These approaches have a lot of potential to speed up the detection of incidents and identification of root causes, however, the models are difficult to train due to imbalanced data and the need to account for interactions between components (Chen et al., 2019).

**References**

Chen, Y., Yang, X., Lin, Q., Zhang, H., Gao, F., Xu, Z., Dang, Y., Zhang, D., Dong, H., Xu, Y., Li, H. and Kang, Y. (2019) ‘Outage Prediction and Diagnosis for Cloud Service Systems’, in The World Wide Web Conference. WWW ’19: The Web Conference, San Francisco CA USA: ACM, pp. 2659–2665. DOI: https://doi.org/10.1145/3308558.3313501.

Jin, P., Zhang, S., Ma, M., Li, H., Kang, Y., Li, L., Liu, Y., Qiao, B., Zhang, C., Zhao, P., He, S., Sarro, F., Dang, Y., Rajmohan, S., Lin, Q. and Zhang, D. (2023) ‘Assess and Summarize: Improve Outage Understanding with Large Language Models’, in Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, San Francisco CA USA: ACM, pp. 1657–1668. DOI https://doi.org/10.1145/3611643.3613891.

Uptime Institute (2022) Uptime Institute’s 2022 Outage Analysis Finds Downtime Costs and Consequences Worsening as Industry Efforts to Curb Outage Frequency Fall Short, Uptime Institute. Available at: https://uptimeinstitute.com/about-ui/press-releases/2022-outage-analysis-finds-downtime-costs-and-consequences-worsening [Accessed: 6 November 2024].

Wang, Y., Li, G., Wang, Z., Kang, Y., Zhou, Y., Zhang, H., Gao, F., Sun, J., Yang, L., Lee, P., Xu, Z., Zhao, P., Qiao, B., Li, L., Zhang, X. and Lin, Q. (2021) ‘Fast Outage Analysis of Large-scale Production Clouds with Service Correlation Mining’. arXiv. Available at: http://arxiv.org/abs/2103.03649 [Accessed: 6 November 2024].