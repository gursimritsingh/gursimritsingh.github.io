---
layout: post
title: Exploring And Visualizing Climates Of Various Countries
---

In this Python tutorial we'll use SQL to query a database we create along with making graphic visuals of the data using Plotly!

## Creating our Database

Lets first make sure we have loaded in the necessary python packages for this.

```python
#packages needed
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sqlite3
from plotly.io import write_html
```

Now lets create our database! 

```python
#creating the database
conn = sqlite3.connect
```

We have just created an empty database. In order for that database to actually be useful to us we need to load in the appropriate data.

```python
#reading in the csv files
#reading in our temperature data in chunks
temps = pd.read_csv("temps_stacked.csv", chunksize = 100000)
#writing to the temperatures table in our database. first we will iterate through to create a FIPS code column which
#will help us later on when querying
for temp in temps:
    temp["FIPS"] = temp["ID"].str[0:2]
    temp.to_sql("temperatures", conn, if_exists = "append", index = False)

#reading in our stations data
stations_url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv"
stations = pd.read_csv(stations_url)
stations.to_sql("stations", conn, if_exists = "replace", index = False)

#reading in our countries data and renaming some columns which will come in handy later
countries_url = "https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv"
countries = pd.read_csv(countries_url)
countries = countries.rename(columns = {"FIPS 10-4" : "FIPS", "Name": "Country"})
countries.to_sql("countries", conn, if_exists = "replace", index = False)

```

Congrats! We have just created out database. Now lets check to see if we did it correctly.

```python
#checking our tables in our newly created database
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())

#checking to see if we populated the database correctly
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
for result in cursor.fetchall():
    print(result[0])
conn.close()
```

## Making a climate database querying function

Now that we have made our database how can we access certain data that we want? Well with this querying function we are able to do just that.
We'll be making a querying function that returns a panda dataframe telling us the temperature of the selected country and much more.

```python
def query_climate_database(country, year_begin, year_end, month):
    """
    This function takes in a country, a starting year, ending year, and month.
    It then returns a pandas dataframe containg information such as station names and temperatures for the selected country.
    """
    conn = sqlite3.connect("climate.db")
    cmd = \
    """
    SELECT S.name, S.latitude, S.longitude, C.country, T.id, T.month, T.year, T.temp
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    LEFT JOIN countries C ON T.fips = C.fips
    WHERE C.country == '{0}' AND T.year >= {1} AND T.year <= {2} AND T.month == {3}
    """.format(country, year_begin, year_end, month)
    climate = pd.read_sql_query(cmd, conn)
    climate.reset_index()
    conn.close()
    return climate
```

Lets see if we did this correctly with a test.

```python
india = query_climate_database(country = "India",
                       year_begin = 1980,
                       year_end = 2020,
                       month = 1)
india.head()
```

![alt text](https://i.gyazo.com/81143a36ad0b42a3f57ba54836afca99.png)

## Making a geographic scatter function

Now lets try making a geographic scatterplot function to better visualize our data on a map using plotly. For this we'll need to import 
two more packages. Using the linear regression package from the sci-kit learn library we can also get the coefficients for the average
change in temperature. Lets get started.

```python
#the necessary modules
from sklearn.linear_model import LinearRegression
import plotly.express as px
import calendar

#our linear regression function
def coef(data_group):
    x = data_group[["Year"]] # 2 brackets because X should be a df
    y = data_group["Temp"]   # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]

def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    """
    this function takes in a given country, starting year, ending year, month, minimum required years, and additional arguments for plot customization.
    it returns an interactive plotly data visualization with labels, titles, and more interactive information.
    """
    #get the relevant dataframe from specified country, year span, and month
    df = query_climate_database(country, year_begin, year_end, month)
    #filter out stations with fewer years of data than min_obs
    df['num_obs'] = df.groupby('NAME')['Year'].transform(len)
    df = df[df['num_obs'] >= min_obs]
    #create column 'Estimated Yearly Increase' containing slope of linear regression
    coefs = df.groupby(["NAME", "Month","LATITUDE","LONGITUDE"]).apply(coef)
    coefs = coefs.reset_index()
    coefs.rename(columns={0: 'Estimated Yearly Increase'}, inplace = True)
    coefs = coefs.round(4)
    #make figure
    fig = px.scatter_mapbox(coefs,
                        lat = "LATITUDE", 
                        lon = "LONGITUDE",
                        hover_name = "NAME",
                        color = "Estimated Yearly Increase",
                        color_continuous_midpoint = 0,
                        title = "Estimates of yearly temperature change in "+ calendar.month_name[month] \
                            +" for stations in "+ country +", years "+str(year_begin) + " - "+ str(year_end),
                        **kwargs)
    return fig
```

Now that we've done that lets try it out by getting the yearly temperature changes in January from India between 1980 and 2020.

```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("India", 1980, 2020, 1,
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)

fig.show()
write_html(fig, "india_plotly.html")
```

{% include india_plotly.html %}

Since this is a function we created we can see this interactive geographic scatterplot with any country and time frame of our choosing.

{% include china_plotly.html %}