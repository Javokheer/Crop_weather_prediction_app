from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
import sklearn as sk
import joblib
from sklearn.preprocessing import LabelEncoder
import time
st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

st.title("Projected impacts of climate change on your local temprature and take action today! ")
# video_file = open(
#     'pexels-mikhail-nilov-8318649 (1440p).mp4', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)
x = st.text_input("How can I call you? An Inspiring Environmental Activist!")

if x:  # This will check if 'x' is not empty
    st.subheader(
        f"Dear {x}, Are you concerned about the future of our planet? Do you want to make a positive impact on the world around you?"
    )
_LOREM_IPSUM = """
It's time to take action on one of the most pressing issues of our time: climate change.

Climate change affects us all, from the air we breathe to the food we eat, and its impacts are becoming increasingly evident in our everyday lives. But there is hope. By raising awareness and taking collective action, we can mitigate its effects and build a more sustainable future for generations to come.

At Group 8 of SP Jain School of Global Management, we're dedicated to promoting knowledge and awareness about climate change and empowering individuals like you
to make a difference. Our interactive platform offers a wealth of resources, tools, and opportunities for you to get involved:

- Explore educational content to deepen your understanding of climate science and its impacts.
- Calculate your carbon footprint and discover practical tips for reducing it in your daily life.
- Connect with a community of like-minded individuals, share ideas, and join forces to tackle climate challenges together.
- Stay informed with the latest news, updates, and success stories from the front lines of climate action.
- Take meaningful action through volunteer opportunities, advocacy campaigns, and sustainable initiatives.
Together, we can be part of the solution. Join us in the fight against climate change and together, let's create a more resilient and sustainable future for all!
"""


def stream_data():
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.02)


if st.button("Message from Authors"):
    st.write_stream(stream_data())



# Using markdown to create a link that looks like a button
st.markdown("""
    <a style='display: block; text-align: center; background-color: #4CAF50; color: white; padding: 14px 20px; margin: 10px 0; border: none; cursor: pointer; width: 100%;' 
    href='https://scholar.google.com/citations?user=y-vi274AAAAJ&hl=en' target='_blank'>Author Profile</a>
    """, unsafe_allow_html=True)


st.image('https://climate.nasa.gov/internal_resources/2710/Effects_page_triptych.jpeg',
         caption='Climate Change Consequences')

import pandas as pd

# Your original DataFrame
data_df = pd.DataFrame({
    "articles": [
        "Financial Assessment under Climate Change",
        "Urban Planning",
        "Mitigation due to Climate Change",
        "Climate Change Finance for Business",
    ],
    "author": [
        "https://www.mdpi.com/1911-8074/15/11/542",
        "https://www.mdpi.com/2413-8851/7/2/46",
        "https://www.mdpi.com/1660-4601/19/19/12233",
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4351342",
    ],
})

# Convert author URLs to HTML links
data_df['author'] = data_df['author'].apply(lambda x: f"<a href='{x}' target='_blank'>Open Reading</a>")

# Display the DataFrame as HTML
st.write(data_df.to_html(escape=False), unsafe_allow_html=True)
st.write("Data that has been used to train the predictive model:")
df = pd.read_csv("Projected_impacts_datasheet_11.24.2021.csv")
st.dataframe(df)

chart_data = df['Local delta T from 2005']
# st.bar_chart(chart_data)
st.line_chart(chart_data)

