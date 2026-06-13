---
title: ePortfolio
---

<h2>Browse Posts by Category</h2>
{% for category in site.categories %}
<details><summary><h3 style="display:inline">{{category[0]}}</h3></summary>
<ul>
    {% for post in category[1] %}
    <li><a href="{{post.url | relative_url }}">{{post.title}}</a></li>
    {% endfor %}
</ul>
</details>
{% endfor %}
