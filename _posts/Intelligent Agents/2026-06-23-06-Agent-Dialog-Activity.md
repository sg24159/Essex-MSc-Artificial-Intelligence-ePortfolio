---
title: "06 Activity: Creating Agent Dialogues"
category: Intelligent Agents
---

Alice and Bob are agents who communicate with KQML (knowledge query and manipulation language) and KIF (knowledge interchange format). This dialog shows how Alice, the procurement agent, would ask Bob, the warehouse agent, about his stock of 50-inch televisions.

`stream-about` is used to request a stream of reponses.

**Request:**

```kqml
(stream-about 
     :sender Alice 
     :receiver Bob
     :language KIF 
     :ontology warehouse 
     :reply-with q1 
     :content ((val (available_stock tv50in)) (val (num_hdmi_ports tv50in)))
)
```

**Response:**
```kqml
(tell
      :sender Bob
      :receiver Alice
      :in-reply-to q1
      :content (= (available_stock tv50in) 19)
)

(tell
     :sender Bob
     :receiver Alice
     :in-reply-to q1
     :content (= (num_hdmi_ports tv50in) 3)
)

(eos
     :sender Bob
     :receiver Alice
     :in-reply-to q1
)
```
