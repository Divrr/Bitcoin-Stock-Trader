import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import dash
from dash import dcc, html, Input, Output
import base64
from algo_run import run_igwo, run_aco

app = dash.Dash(__name__)
app.title = "Bitcoin Trading Bot Optimizer"

app.layout = html.Div([
    html.Div([
        html.H1("Bitcoin Trading Bot Optimizer", className="title"),
        html.P("Select an algorithm and visualize the optimized trading signals.", className="subtitle"),

        html.Div([
            html.Label("Select Optimization Algorithm:"),
            dcc.Dropdown(
                id="algo-dropdown",
                options=[
                    {"label": "IGWO (Improved Grey Wolf Optimizer)", "value": "IGWO"},
                    {"label": "ACO (Ant Colony Optimization)", "value": "ACO"}
                ],
                value="IGWO",
                style={"width": "100%"}
            ),
            html.Button("Run Optimization", id="run-button", className="button"),
        ], className="input-section"),

        dcc.Loading(
            id="loading",
            type="circle",
            children=[
                html.Div(id="output-text", className="status"),
                html.Img(id="signal-plot", className="image")
            ]
        )
    ], className="container")
], className="main")

@app.callback(
    [Output("output-text", "children"),
     Output("signal-plot", "src")],
    Input("run-button", "n_clicks"),
    [Input("algo-dropdown", "value")]
)
def run_and_display(n_clicks, selected_algo):
    if not n_clicks:
        return "", ""

    try:
        if selected_algo == "IGWO":
            plot_path, elapsed_time = run_igwo()
        else:
            plot_path, elapsed_time = run_aco()

        with open(plot_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
        return f"✅ {selected_algo} optimization completed in {elapsed_time:.2f} seconds.", f"data:image/png;base64,{encoded}"

    except Exception as e:
        return f"❌ Error running {selected_algo}: {str(e)}", ""

if __name__ == "__main__":
    app.run(debug=True)
