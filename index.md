---
title: ePortfolio
---

{% for category in site.categories %}
<h3>{{category[0]}}</h3>
<ul>
    {% for post in category[1] %}
    <li><a href="/Essex-MSc-Artificial-Intelligence-ePortfolio{{post.url}}">{{post.title}}</a></li>
    {% endfor %}
</ul>
{% endfor %}