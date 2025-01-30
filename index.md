---
title: ePortfolio
---

<h2>Browse Posts by Category</h2>
{% for category in site.categories %}
<h3>{{category[0]}}</h3>
<ul>
    {% for post in category[1] %}
    <li><a href="{{post.url | relative_url }}">{{post.title}}</a></li>
    {% endfor %}
</ul>
{% endfor %}