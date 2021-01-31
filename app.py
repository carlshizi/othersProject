import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# COVID TABLE -- SOURCE
df_covid = pd.read_csv('data/node_245_field_data_table_und_0.csv')
# Data List
df_A = df_covid['County or Counties Served'].tolist()
df_B = df_covid['Organization'].tolist()
df_C = (df_covid['Website']).str.replace('<a/>', '</a>').tolist()
df_D = (df_covid['Phone Number'].fillna('')).tolist()
df_E = (df_covid['Email'].fillna('')).tolist()
df_F = (df_covid['More Information'].fillna('')).str.replace('<a/>', '</a>')
df_Column = df_covid.columns.tolist()

headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

# COVID TABLE -- Figure
fig_covid = go.Figure(data=[go.Table(
  header=dict(
    values=df_Column,
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left', 'center'],
    font=dict(color='white', size=12)
  ),
  cells=dict(
    values=[
      df_A, df_B, df_C, df_D, df_E, df_F],
    line_color='darkslategray',
    # 2-D list of colors for alternating rows
    fill_color=[[rowOddColor, rowEvenColor, rowOddColor, rowEvenColor, rowOddColor]*5],
    align=['left', 'center'],
    font=dict(color='darkslategray', size=11)
    ))
])
fig_covid.update_layout(
    title="List of Covid Vaccine Providers in NC"
)


# HISTOGRAM -- SOURCE
df = pd.read_csv('data/Estimated YLL.csv')

df['Unnamed: 3'] = df['Unnamed: 3'].str.replace(',', '')
df['Unnamed: 4'] = df['Unnamed: 4'].str.replace(',', '')

df.rename(columns={"Estimated Years of Life Lost (YLL) due to Diabetes, United States, 2013": "Sex",
                   "Unnamed: 1": "Age Group (in years)", "Unnamed: 2": "Average YLLs due to Diabetes",
                   "Unnamed: 3": "Number of Persons with Diabetes (in thousands)",
                   "Unnamed: 4": "Total YLLs due to Diabetes (in thousands)"}, inplace=True)

df.drop((df.index[[0]]), inplace=True)
del df['Sex']
df.insert(1, 'Sex', ["Total", "Total", "Total", "Total", "Total",
                     "Males", "Males", "Males", "Males", "Males",
                     "Females", "Females", "Females", "Females", "Females"])

# HISTOGRAM -- Dataframe with omitted column
df2 = df.iloc[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]]

agemod = "Age Group (in years)"
Nupmod = "Number of Persons with Diabetes (in thousands)"
TotYLLmod = "Total YLLs due to Diabetes (in thousands)"
Averagemod = "Average YLLs due to Diabetes"

fig_hist = px.histogram(df2, x=agemod, y=TotYLLmod,
                        range_y=[0, 130_000], color='Sex',
                        hover_data=df2.columns, title="Total Year of Life Lost Due to Diabetes")


# PIE CHART -- SOURCE
# Dataframe3 Query only total amount
df3 = df2.query("Sex == 'Total'")

# PIE CHART -- Pie Chart Figure
fig_pie = px.pie(df3, values='Number of Persons with Diabetes (in thousands)',
                 names='Age Group (in years)',
                 title='Percentage of the Population with Diabetes (Per Age Group)',
                 )
fig_pie.update_traces(textposition='inside', textinfo='percent')
fig_pie.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')


# SCATTER PLOT -- LINEAR REGRESSION SOURCE
df_lr = px.data.tips()
X = df_lr.total_bill[:, None]
X_train, X_test, y_train, y_test = train_test_split(X, df_lr.tip, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

x_range = np.linspace(X.min(), X.max(), 120)
y_range = model.predict(x_range.reshape(-1, 1))

# SCATTER PLOT -- Figure
fig_lr = go.Figure([
    go.Scatter(x=X_train.squeeze(), y=y_train, name='Train', mode='markers'),
    go.Scatter(x=X_test.squeeze(), y=y_test, name='Test Set', mode='markers'),
    go.Scatter(x=x_range, y=y_range, name='Predicted Value')
])
fig_lr.update_layout(
    title='Scatter Plot with Linear Regression Function'
)


# App Start
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI])
server = app.server

# Style for Sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# Style for Main
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Examples", className="display-6"),
        html.Hr(),
        html.P(
            "A collection of different Plotly graphs"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Covid Table", href="/", active="exact"),
                dbc.NavLink("Histogram", href="/page-1", active="exact"),
                dbc.NavLink("Pie Chart", href="/page-2", active="exact"),
                dbc.NavLink("Scatterplot", href="/page-3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)


# APP LAYOUT
app.layout = dbc.Container(html.Div(
    # Header: includes logo, title, and clickable links
    # Main Title
    [
        dbc.Row(dbc.Col(style={'padding': 30})),
        dbc.Row(
            [
                dbc.Col(html.Div(html.Img(src='static/Kola Title.png',
                                          height='50',
                                          width='true'),
                                 style={
                                     'text-align': 'right',
                                     'position': 'relative',
                                     'left': 120,
                                     'bottom': 0
                                       }
                                 )),

                # Header: Clickable Images
                dbc.Col(html.Div([
                    html.A([
                        html.Img(
                            src='static/Email_White.png',
                            style={
                                'height': '8%',
                                'width': '8%',
                                'float': 'right',
                                'position': 'relative',
                                'padding-top': 0,
                                'padding-right': 0,
                                'left': 0,
                                'bottom': 0})
                    ], href='mailto:carlfoshizi@gmail.com'),
                    html.A([
                        html.Img(
                            src='static/LinkedIn_White.png',
                            style={
                                'height': '9%',
                                'width': '9%',
                                'float': 'right',
                                'position': 'relative',
                                'padding-top': 0,
                                'padding-right': 0,
                                'right': 20,
                                'bottom': 4})
                    ], href='https://www.linkedin.com/in/kola-ladipo/'),
                    html.A([
                        html.Img(
                            src='static/Github_White.png',
                            style={
                                'height': '8%',
                                'width': '8%',
                                'float': 'right',
                                'position': 'relative',
                                'padding-top': 0,
                                'padding-right': 0,
                                'right': 40,
                                'bottom': 0})
                    ], href='https://github.com/carlshizi?tab=repositories/')
                ]))
            ]
        ),


        # Subtitle
        dbc.Row(dbc.Col(html.Div([dcc.Location(id="url"), sidebar, content]))
                ),


        # Spacing
        dbc.Row(dbc.Col(html.Div(style={'padding': 20}))),
    ]),
    fluid=False
)

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([dbc.Card(dcc.Graph(id="Covid Table",
                                  figure=fig_covid)),
                         dbc.Row(dbc.Col(style={'padding': 10})),
                         dbc.Card(dbc.CardBody(
                             [
                                 html.H5("Covid Table",
                                         className="card-title"),
                                 html.P(
                                     "This is a simple table with alternating row colors."
                                     " I 'cleaned' the data to fix improperly formatted"
                                     ".html tags.",
                                     className="card-text",
                                 ),
                                 html.P(dcc.Markdown(
                                     '''Data Source – NC Department of Health and Human Services''',
                                     className="card-text"),
                                 ),
                                 html.P(dcc.Markdown(
                                     '''Code Snippet on Jupyter – [https://github.com/carlshizi/othersProject/blob/master/covid_table.ipynb]
                                     (https://github.com/carlshizi/othersProject/blob/master/covid_table.ipynb)''',
                                     className="card-text"),
                                 ),
                             ]
                         ))])
    elif pathname == "/page-1":
        return html.Div([dbc.Card(dcc.Graph(id="Histogram",
                                  figure=fig_hist)),
                         dbc.Row(dbc.Col(style={'padding': 10})),
                         dbc.Card(dbc.CardBody(
                             [
                                 html.H5("Histogram", className="card-title"),
                                 html.P(
                                     "Estimated Years of Life Lost (YLL) due to Diabetes, United States, 2013."
                                     " I did a bit of data wrangling to restructure the data for graphing.",
                                     className="card-text",
                                 ),
                                 html.P(dcc.Markdown(
                                     '''Data Source - The Centers for Disease Control and Prevention''',
                                     className="card-text"),
                                 ),
                                 html.P(dcc.Markdown(
                                     '''Code Snippet on Jupyter – [https://github.com/carlshizi/othersProject/blob/master/covid_table.ipynb]
                                     (https://github.com/carlshizi/othersProject/blob/master/covid_table.ipynb)''',
                                     className="card-text"),
                                 ),
                             ]
                         ))])
    elif pathname == "/page-2":
        return html.Div([dbc.Card(dcc.Graph(id="Pie",
                                  figure=fig_pie)),
                         dbc.Row(dbc.Col(style={'padding': 10})),
                         dbc.Card(dbc.CardBody(
                             [
                                 html.H5("Pie Chart", className="card-title"),
                                 html.P(
                                     "The CDC Diabetes dataset plotted with a pie chart",
                                     className="card-text",
                                 ),
                                 html.P(dcc.Markdown(
                                     '''Code Snippet on Jupyter – [https://github.com/carlshizi/othersProject/blob/master/DIabetes%20PieChart.ipynb]
                                     (https://github.com/carlshizi/othersProject/blob/master/DIabetes%20PieChart.ipynb)''',
                                     className="card-text"),
                                 ),
                             ]
                         ))])
    elif pathname == "/page-3":
        return html.Div([dbc.Card(dcc.Graph(id="Linear Regression",
                                  figure=fig_lr)),
                         dbc.Row(dbc.Col(style={'padding': 10})),
                         dbc.Card(dbc.CardBody(
                             [
                                 html.H5("Machine Learning", className="card-title"),
                                 html.P(
                                     "Using scikit-learn, numpy, and plotly for prediction.",
                                     className="card-text",
                                 ),
                                 html.P(dcc.Markdown(
                                     '''Data Source: `px.data.tips`''',
                                     className="card-text"),
                                 ),
                             ]
                         ))])
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server()
