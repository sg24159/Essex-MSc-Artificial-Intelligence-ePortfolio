---
title: ePortfolio
---

{% assign newest_date = 0 %}
{% for category in site.categories %}
    {% assign latest_post = category[1] | first %}
    {% assign latest_date = latest_post.date | date: '%Y%m%d' | plus: 0 %}
    {% if latest_date > newest_date %}
        {% assign newest_date = latest_date %}
        {% assign newest_category = category[0] %}
    {% endif %}
{% endfor %}

<h2 class="post-list-heading">Browse Posts by Category</h2>
{% for category in site.categories %}
<details{% if category[0] == newest_category %} open{% endif %}><summary><h3 style="display:inline">{{category[0]}}</h3></summary>
<ul>
    {% assign sorted_posts = category[1] | sort: 'title' | reverse %}
    {% for post in sorted_posts %}
    <li><a href="{{post.url | relative_url }}">{{post.title}}</a></li>
    {% endfor %}
</ul>
</details>
{% endfor %}
