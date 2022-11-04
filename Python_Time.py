from inspect import isclass
import streamlit as st
import pandas as pd
from darts import TimeSeries
st.set_page_config(page_title='Darts + Streamlit', page_icon=":dart:", layout='wide')
    
df = pd.read_csv("C:/Users/john.tan/Downloads/air_passengers.csv", delimiter=",")
 
series = TimeSeries.from_dataframe(df, 'Month', '#Passengers') # Create a TimeSeries, specifying the time and value columns
train, val = series[:-36], series[-36:] # Set aside the last 36 months as a validation series

with st.expander("Dataframe View"):
    st.dataframe(series.pd_dataframe())

from darts.models import ExponentialSmoothing
model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples=1000)

import matplotlib.pyplot as plt
fig = plt.figure()
series.plot()
prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
plt.legend()
st.pyplot(fig)

interactive_fig = plt.figure()
series.plot()

st.subheader("Training Controls")
num_periods = st.slider("Number of validation months", min_value=2, max_value=len(series) - 24, value=36, help='How many months worth of datapoints to exclude from training')
num_samples = st.number_input("Number of prediction samples", min_value=1, max_value=10000, value=1000, help="Number of times a prediction is sampled for a probabilistic model")
st.subheader("Plotting Controls")
low_quantile = st.slider('Lower Percentile', min_value=0.01, max_value=0.99, value=0.05, help='The quantile to use for the lower bound of the plotted confidence interval.')
high_quantile = st.slider('High Percentile', min_value=0.01, max_value=0.99, value=0.95, help='The quantile to use for the upper bound of the plotted confidence interval.')
    
train, val = series[:-num_periods], series[-num_periods:]
model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples=num_samples)
prediction.plot(label='forecast', low_quantile=low_quantile, high_quantile=high_quantile)

plt.legend()
st.pyplot(interactive_fig)