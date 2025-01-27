---
title: "06 Group Project: AirBnB Data Preprocessing And Exporation"
category: Machine Learning
---

## 1 Loading the Data
We'll start by loading some useful libraries and the data


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
raw_data = pd.read_csv('../data/AB_NYC_2019.csv')
```

## 2 Checking the Data
Look at a sample of the data, some statistics, and basic plots.
This serves to guide our cleaning process and note any relationships we should be aware of later.


```python
raw_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>2787</td>
      <td>John</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>Private room</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2595</td>
      <td>Skylit Midtown Castle</td>
      <td>2845</td>
      <td>Jennifer</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3647</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>4632</td>
      <td>Elisabeth</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>Private room</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3831</td>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>1</td>
      <td>270</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5022</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>7192</td>
      <td>Laura</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(raw_data['neighbourhood_group'].unique())
print(raw_data['room_type'].unique())
```

    ['Brooklyn' 'Manhattan' 'Queens' 'Staten Island' 'Bronx']
    ['Private room' 'Entire home/apt' 'Shared room']



```python
print(raw_data.shape)
print(10052/raw_data.shape[0])
pd.concat([raw_data.isnull().sum(), raw_data.dtypes], axis=1, 
          keys=['Null Count', 'Data Types'])
```

    (48895, 16)
    0.20558339298496778





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Null Count</th>
      <th>Data Types</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>name</th>
      <td>16</td>
      <td>object</td>
    </tr>
    <tr>
      <th>host_id</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>host_name</th>
      <td>21</td>
      <td>object</td>
    </tr>
    <tr>
      <th>neighbourhood_group</th>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>neighbourhood</th>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>room_type</th>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>price</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>minimum_nights</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>last_review</th>
      <td>10052</td>
      <td>object</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>10052</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>0</td>
      <td>int64</td>
    </tr>
  </tbody>
</table>
</div>



`last_review` is a string type, we might prefer to convert it into a numeric type.


```python
raw_data['last_review_date'] = pd.to_datetime(raw_data['last_review'])
raw_data.drop(['last_review'], axis=1).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
      <th>last_review_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.889500e+04</td>
      <td>4.889500e+04</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>38843.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>38843</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.901714e+07</td>
      <td>6.762001e+07</td>
      <td>40.728949</td>
      <td>-73.952170</td>
      <td>152.720687</td>
      <td>7.029962</td>
      <td>23.274466</td>
      <td>1.373221</td>
      <td>7.143982</td>
      <td>112.781327</td>
      <td>2018-10-04 01:47:23.910099456</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.539000e+03</td>
      <td>2.438000e+03</td>
      <td>40.499790</td>
      <td>-74.244420</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2011-03-28 00:00:00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.471945e+06</td>
      <td>7.822033e+06</td>
      <td>40.690100</td>
      <td>-73.983070</td>
      <td>69.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.190000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2018-07-08 00:00:00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.967728e+07</td>
      <td>3.079382e+07</td>
      <td>40.723070</td>
      <td>-73.955680</td>
      <td>106.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.720000</td>
      <td>1.000000</td>
      <td>45.000000</td>
      <td>2019-05-19 00:00:00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.915218e+07</td>
      <td>1.074344e+08</td>
      <td>40.763115</td>
      <td>-73.936275</td>
      <td>175.000000</td>
      <td>5.000000</td>
      <td>24.000000</td>
      <td>2.020000</td>
      <td>2.000000</td>
      <td>227.000000</td>
      <td>2019-06-23 00:00:00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.648724e+07</td>
      <td>2.743213e+08</td>
      <td>40.913060</td>
      <td>-73.712990</td>
      <td>10000.000000</td>
      <td>1250.000000</td>
      <td>629.000000</td>
      <td>58.500000</td>
      <td>327.000000</td>
      <td>365.000000</td>
      <td>2019-07-08 00:00:00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.098311e+07</td>
      <td>7.861097e+07</td>
      <td>0.054530</td>
      <td>0.046157</td>
      <td>240.154170</td>
      <td>20.510550</td>
      <td>44.550582</td>
      <td>1.680442</td>
      <td>32.952519</td>
      <td>131.622289</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



From the above, we can see a few interesting things to note. Aside from `name` and `host_name`, the text fields appear to be formatted consistently.
The interquartile differences for columns related to price, nights and reviews indicate a positively skewed distribution with significant outliers.
There are a few null values for `name` and `host_name`, those rows could be dropped, however, they are unlikely to be useful for our analysis. Therefore, dropping the columns is also an option. The columns for `last_review` and `reviews_per_month` are roughtly 20% null. There are some outdated listings in this data set as well based on the minimum of the `last_review_date` column.

Next, look for relationships that could inform further cleaning decisions. For example, the 25th percentile for `number_of_reviews` is 1, suggesting that null `last_review` and `reviews_per_month` could be related to having no reviews.


```python
raw_data[raw_data['number_of_reviews'] == 0].isnull().sum()
```




    id                                    0
    name                                 10
    host_id                               0
    host_name                             5
    neighbourhood_group                   0
    neighbourhood                         0
    latitude                              0
    longitude                             0
    room_type                             0
    price                                 0
    minimum_nights                        0
    number_of_reviews                     0
    last_review                       10052
    reviews_per_month                 10052
    calculated_host_listings_count        0
    availability_365                      0
    last_review_date                  10052
    dtype: int64



It is clear that having no reviews fully explains null entries for `last_review` and `reviews_per_month`, it also might be related to some of the null name values.


```python
numeric_data = raw_data.select_dtypes(include=[np.number, np.datetime64])

# Using Spearman correlation, we have not yet removed outliers or tested for normality.
sns.heatmap(numeric_data.corr(method='spearman'), annot=True, cmap='vlag', 
            linewidths=0.8, vmin=-1, vmax=1)
```




    <Axes: >




    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/data_preprocessing_11_1.png)
    


There are some interesting relationships to note here, `id` and `host_id` weakly correlate, suggesting that they are sequential, and could be used as a proxy for the age of the listing or host.
The relationship of `number_of_reviews` with `last_review` and `reviews_per_month` is moderate, so removing the null-filled `last_review` and `reviews_per_month` columns may be reasonable. Most machine learning approaches work best with uncorrelated features.


## 3 Cleaning the Data
In the above section, we made note of several issues with the data that need to be addressed:

- `last_review` should be a numeric type. It also has null values without an obvious default value. It will be dropped.
- `name` and `host_name` contain null values and are not easy to process into features. They can be removed.
- ID values are usually noise, but in this case, they could be used to engineer a feature approximately representing a listing's age. 
Unless later analysis determines a need to group by host or approximate age, these columns can be dropped.
- `reviews_per_month` contains null values, with 0 being an obvious choice of default value.

During the initial exploration, `last_review_date` was created. It could be used to filter outdated listings with comparison operators if the null values are handled. It will be dropped unless there is a reason to filter older listings as well as drop the null values.


```python
sns.histplot(raw_data.dropna()['last_review_date'])
```




    <Axes: xlabel='last_review_date', ylabel='Count'>




    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/data_preprocessing_14_1.png)
    



```python
data = raw_data.drop(['id', 'name', 'host_id', 'host_name', 'last_review',
                      'last_review_date'], axis=1)
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)
data.isnull().sum()
```




    neighbourhood_group               0
    neighbourhood                     0
    latitude                          0
    longitude                         0
    room_type                         0
    price                             0
    minimum_nights                    0
    number_of_reviews                 0
    reviews_per_month                 0
    calculated_host_listings_count    0
    availability_365                  0
    dtype: int64




```python
sns.boxplot(data, orient='h')
```




    <Axes: >




    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/data_preprocessing_16_1.png)
    


There are some significant outliers in the data. `price` has long tail that will cause problems with scaling, it also has some zero values. `minimum_nights` has some values that are greater than a year or more, and therefore do not represent the short term rental market. 


```python
data_cleaned  = data[(data['minimum_nights'] < 30) 
                     & (data['price'] > 20) 
                     & (data['price'] < 300)  
                     ].copy()

sns.boxplot(data_cleaned, orient='h')
```




    <Axes: >




    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/data_preprocessing_18_1.png)
    


This has reduced the number of outlier listings, but there are still some long tails. The negative longitude values will also cause issues with some algorithms, however  scaling will address it.


```python
from sklearn.preprocessing import scale, minmax_scale
numeric_data = data_cleaned.select_dtypes(include=[np.number, np.datetime64])
sns.boxplot(minmax_scale(numeric_data), orient='h')
```




    <Axes: >




    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/data_preprocessing_20_1.png)
    


This scaling test indicates that outliers are still an issue with `calculated_host_listings_count`, `reviews_per_month`, `number_of_reviews` and `minimum_nights`. This will be kept in mind for processing the data

# 4 Augmenting the Data
*This section was contributed by another team member*

Converting the latitude & longitude to the distance (in KM) from the famous NYC landmarks


```python
# Landmark coordinates
landmarks = {
    "Times Square": (40.7580, -73.9855),
    "Statue of Liberty": (40.6892, -74.0445),
    "Central Park": (40.7851, -73.9683),
    "Empire State Building": (40.7488, -73.9854),
    "Brooklyn Bridge": (40.7061, -73.9969),
    "One World Trade Center": (40.7128, -74.0131),
    "Metropolitan Museum of Art": (40.7794, -73.9632),
}

# Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Add distances to landmarks
for landmark, coords in landmarks.items():
    data_cleaned[f"distance_to_{landmark.replace(' ', '_')}"] = data_cleaned.apply(
        lambda row: haversine(row['latitude'], row['longitude'], coords[0], coords[1]), axis=1
    )
```


```python
data_preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>distance_to_Times_Square</th>
      <th>distance_to_Statue_of_Liberty</th>
      <th>distance_to_Central_Park</th>
      <th>distance_to_Empire_State_Building</th>
      <th>distance_to_Brooklyn_Bridge</th>
      <th>distance_to_One_World_Trade_Center</th>
      <th>distance_to_Metropolitan_Museum_of_Art</th>
      <th>availability_percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>Private room</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>0.21</td>
      <td>6</td>
      <td>12.337898</td>
      <td>7.649799</td>
      <td>15.305378</td>
      <td>11.318587</td>
      <td>6.837559</td>
      <td>8.033374</td>
      <td>14.688071</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>1</td>
      <td>40</td>
      <td>0.38</td>
      <td>2</td>
      <td>0.508366</td>
      <td>8.803656</td>
      <td>3.734987</td>
      <td>0.553268</td>
      <td>5.398568</td>
      <td>5.168139</td>
      <td>3.349388</td>
      <td>97.260274</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>Private room</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
      <td>6.757240</td>
      <td>15.881167</td>
      <td>3.465981</td>
      <td>7.632440</td>
      <td>12.346239</td>
      <td>12.265111</td>
      <td>3.750044</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>1</td>
      <td>40</td>
      <td>4.64</td>
      <td>1</td>
      <td>8.387034</td>
      <td>7.159264</td>
      <td>11.138311</td>
      <td>7.401157</td>
      <td>3.903320</td>
      <td>5.447904</td>
      <td>10.485241</td>
      <td>53.150685</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>0.10</td>
      <td>1</td>
      <td>5.701496</td>
      <td>14.813350</td>
      <td>2.532135</td>
      <td>6.535489</td>
      <td>11.200439</td>
      <td>11.167655</td>
      <td>2.670366</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_cleaned, x='neighbourhood_group', y='distance_to_Times_Square')
plt.title("Distance to Times Square by Neighbourhood Group")
plt.xlabel("Neighbourhood Group")
plt.ylabel("Distance to Times Square (km)")
plt.show()
```


    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/data_preprocessing_25_0.png)
    



Finally, we convert the total days available to the percentage for an easier interpretation.


```python
data_cleaned['availability_percentage'] = (data_cleaned['availability_365'] / 365) * 100
```


```python
data_cleaned = data_cleaned.drop(columns=['availability_365'])
data_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>distance_to_Times_Square</th>
      <th>distance_to_Statue_of_Liberty</th>
      <th>distance_to_Central_Park</th>
      <th>distance_to_Empire_State_Building</th>
      <th>distance_to_Brooklyn_Bridge</th>
      <th>distance_to_One_World_Trade_Center</th>
      <th>distance_to_Metropolitan_Museum_of_Art</th>
      <th>availability_percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>Private room</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>0.21</td>
      <td>6</td>
      <td>12.337898</td>
      <td>7.649799</td>
      <td>15.305378</td>
      <td>11.318587</td>
      <td>6.837559</td>
      <td>8.033374</td>
      <td>14.688071</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>0.38</td>
      <td>2</td>
      <td>0.508366</td>
      <td>8.803656</td>
      <td>3.734987</td>
      <td>0.553268</td>
      <td>5.398568</td>
      <td>5.168139</td>
      <td>3.349388</td>
      <td>97.260274</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>Private room</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
      <td>6.757240</td>
      <td>15.881167</td>
      <td>3.465981</td>
      <td>7.632440</td>
      <td>12.346239</td>
      <td>12.265111</td>
      <td>3.750044</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>1</td>
      <td>270</td>
      <td>4.64</td>
      <td>1</td>
      <td>8.387034</td>
      <td>7.159264</td>
      <td>11.138311</td>
      <td>7.401157</td>
      <td>3.903320</td>
      <td>5.447904</td>
      <td>10.485241</td>
      <td>53.150685</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>0.10</td>
      <td>1</td>
      <td>5.701496</td>
      <td>14.813350</td>
      <td>2.532135</td>
      <td>6.535489</td>
      <td>11.200439</td>
      <td>11.167655</td>
      <td>2.670366</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_cleaned.to_csv('../data/cleaned_data.csv', index=False)
```

## 5 Preproccessing
As shown previously, there are still some outliers affecting the quality of the data, however, they are not extreme enough to drop.
The simplest solution is to set a ceiling for them.


```python
data_preprocessed = data_cleaned.copy()
data_preprocessed['number_of_reviews'] = data_preprocessed[
    'number_of_reviews'].clip(upper=40)
data_preprocessed['reviews_per_month'] = data_preprocessed[
    'reviews_per_month'].clip(upper=10)
data_preprocessed['calculated_host_listings_count'] = data_preprocessed[
    'calculated_host_listings_count'].clip(upper=10)

numeric_data = data_preprocessed.select_dtypes(include=[np.number, np.datetime64]
                ).drop(['price'], axis=1) # This column is in an acceptable state and can be ignored for now

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=numeric_data, orient='h', ax=axes[0])
axes[0].set_title('Unscaled Data')
scaled_data = minmax_scale(numeric_data)
sns.boxplot(data=scaled_data, orient='h', ax=axes[1])
axes[1].set_title('Scaled Data')
plt.tight_layout()
```


    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/data_preprocessing_31_0.png)
    



```python
data_preprocessed.to_csv('../data/processed_data.csv', index=False)
```
