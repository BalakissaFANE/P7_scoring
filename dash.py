import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import requests
import plotly.express as px
from dash import Dash, dcc, html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
feats = requests.get("http://127.0.0.1:8083/feats/").json()

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                feats,
                'EXT_SOURCE_3',
                id='crossfilter-feature',
            )
        ],
            style={'width': '66%', 'display': 'inline-block'}),

        html.Div([
            dcc.Input(id="client_id", type="number", value='100003'),
            html.Button(id="bouton_val", n_clicks=0, children="Valider")
        ],
            style={'width': '33%', "float": "right", 'display': 'inline-block'})
    ]),

    html.Div(dcc.Graph(id='bar_mean', figure={"layout": {"height": 800}}),
             style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align": "middle"}),
    html.Div(dcc.Graph(id='boxplot', figure={"layout": {"height": 800}}),
             style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align": "middle"}),
    html.Div([
        dcc.Graph(id='score'),
        dcc.Graph(id='feature_importance')
    ], style={'display': 'inline-block', 'width': '33%', "float": "right"}),
    html.Div([
        dcc.Graph(id='age')
    ], style={'display': 'inline-block', 'width': '33%', "float": "right"})
])


@app.callback(Output('score', 'figure'),
              Input('bouton_val', 'n_clicks'),
              State('client_id', 'value'))
def update_score(n_clicks, client_id):
    score_min = requests.get("http://127.0.0.1:8083/score_min/").json()["score_min"] * 100

    r = requests.get("http://127.0.0.1:8083/predictions", params={"client_id": client_id})
    val = r.json()["prediction"] * 100
    if val > score_min:
        accept = "Accepté"
        color = "darkgreen"
    else:
        accept = "Refusé"
        color = "darkred"
    fig1 = go.Figure()

    fig1.add_trace(go.Indicator(
        domain={"x": [0, 1], "y": [0, 1]},

        title={"text": "Score", "font_size": 40},
        value=val,
        number={"font_size": 50},

        mode="gauge + number",

        gauge={
            "shape": "angular",
            "steps": [
                {"range": [0, score_min], "color": "red"},
                {"range": [score_min, 100], "color": "green"}
            ],
            "bar": {"color": "black", "thickness": 0.5},
            "axis": {"range": [None, 100]}
        }
    )
    )

    fig1.add_annotation(x=0.5, y=0.4, text=accept, font=dict(size=30, color=color), showarrow=False)

    return fig1


@app.callback(Output('feature_importance', 'figure'),
              Input('bouton_val', 'n_clicks'),
              State('client_id', 'value'))
def update_fi(n_clicks, client_id):
    shap_vals = requests.get("http://127.0.0.1:8083/features_importances", params={"client_id": client_id}).json()

    df_feats = pd.DataFrame(shap_vals, columns=["importances"])
    df_feats["feats"] = feats
    df_feats["abs"] = abs(df_feats["importances"])
    df_feats["Influence"] = np.where(df_feats["importances"] < 0, "Negative", "Positive")
    df_feats.sort_values(by="abs", ascending=False, inplace=True)
    df_feats.drop(columns=["abs"], inplace=True)

    fig2 = px.bar(df_feats.iloc[:10],
                  x="importances",
                  y="feats",
                  color="Influence",
                  orientation="h",
                  title="Principales données influant sur le résultat")
    fig2.update_xaxes(title="Impact sur le résultat")
    fig2.update_yaxes(title="Variable étudiée")

    return fig2


@app.callback(Output('crossfilter-feature', 'value'),
              Input('feature_importance', 'clickData'))
def change_feat(clickdata):
    if clickdata is None:
        return "EXT_SOURCE_3"
    else:
        return clickdata["points"][0]["y"]


@app.callback(Output('bar_mean', 'figure'),
              Input('bouton_val', 'n_clicks'),
              Input("crossfilter-feature", "value"),
              State('client_id', 'value'))
def plot_bar(n_clicks, feature, client_id):
    results = requests.get("http://127.0.0.1:8083/bar",
                           params={"client_id": client_id, "feature": feature}).json()

    fig3 = px.bar(
        x=["client", "moyenne"],
        y=[results[0], results[1]],
        color=[results[0], results[1]],
        title="Comparaison du client à la moyenne")
    fig3.update_xaxes(title="")
    fig3.update_yaxes(title="Valeur")

    return fig3


@app.callback(Output('boxplot', 'figure'),
              Input("crossfilter-feature", "value"))
def plot_box(feature):
    dff = requests.get("http://127.0.0.1:8083/boxplot",
                       params={"feature": feature}).json()

    fig4 = px.box(dff, title="Répartition de la variable dans la clientèle")
    fig4.update_xaxes(title="")
    fig4.update_yaxes(title="Valeur")

    return fig4


@app.callback(Output('age', 'figure'),
              Input('bouton_val', 'n_clicks'),
              State('client_id', 'value'))
def update_score(n_clicks, client_id):
    age_min = requests.get("http://127.0.0.1:8083/age_min/").json()["age_p_min"]
    age_max = requests.get("http://127.0.0.1:8083/age_max/").json()["age_p_max"]
    r = requests.get("http://127.0.0.1:8083/predictions", params={"client_id": client_id})
    val = r.json()["prediction"] * 100
    if val > age_min or val < age_max:
        accept = "Accepté"
        color = "darkgreen"
    else:
        accept = "Refusé"
        color = "darkred"
    fig5 = go.Figure()

    fig5.add_trace(go.Indicator(
        domain={"x": [0, 1], "y": [0, 1]},

        title={"text": "Age", "font_size": 40},
        value=val,
        number={"font_size": 50},

        mode="gauge + number",

        gauge={
            "shape": "angular",
            "steps": [
                {"range": [0, age_min], "color": "red"},
                {"range": [age_max, 100], "color": "green"}
            ],
            "bar": {"color": "black", "thickness": 0.5},
            "axis": {"range": [None, 100]}
        }
    )
    )

    fig5.add_annotation(x=0.5, y=0.4, text=accept, font=dict(size=30, color=color), showarrow=False)

    return fig5


if __name__ == '__main__':
    app.run_server(debug=True)
