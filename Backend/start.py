import pandas as pd 
import plotly.express as px 


df = pd.read_csv('bigdataforyaw.csv')

print(df.head())

df_tech = df[df['internal_industry'] == "Technology, Advertising, Media, and Information"]

tech_count = len(df_tech)

tech_counts_by_market = df_tech['market'].value_counts()

print(tech_counts_by_market)