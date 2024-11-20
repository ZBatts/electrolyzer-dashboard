import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import base64
import io

# Initialize the app
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.SUPERHERO],
)
server = app.server
app.title = "Electrolyzer Performance Dashboard"

# Define the layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        # Header
        dbc.Row(
            [
                dbc.Col(
                    html.H3(
                        "Electrolyzer Performance Dashboard",
                        style={"textAlign": "center", "color": "#fec036"},
                    )
                )
            ],
            justify="center",
        ),
        html.Hr(),
        # File upload
        dbc.Row(
            [
                dbc.Col(
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(["Drag and Drop or ", html.A("Upload CSV")]),
                        style={
                            "width": "100%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                        },
                    ),
                    width=6,
                )
            ],
            justify="center",
        ),
        html.Br(),
        # Main dashboard layout
        dbc.Row(
            [
                # Graphs
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="main-graph",
                                    config={"displayModeBar": False},
                                ),
                                dcc.Dropdown(
                                    id="feature-dropdown",
                                    options=[],
                                    value=[],
                                    multi=True,
                                    placeholder="Select Y-axis Variables",
                                    style={"color": "black"},
                                ),
                            ]
                        ),
                    ),
                    width=8,
                ),
                # Time Column Display
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6("Detected X-axis (Time Column):", style={"color": "#fec036"}),
                                html.Div(id="time-column-display", style={"color": "white"}),
                            ]
                        ),
                    ),
                    width=4,
                ),
            ],
        ),
    ],
)

# Store uploaded data globally
uploaded_data = {}

# Callbacks for uploading and processing data
@app.callback(
    [
        Output("feature-dropdown", "options"),
        Output("time-column-display", "children"),
    ],
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_dropdown(contents, filename):
    global uploaded_data
    if contents is None:
        return [], "No data uploaded"

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    uploaded_data["df"] = df

    # Detect the time column (simple heuristic: look for keywords)
    time_keywords = ["time", "hours", "operating", "date"]
    time_column = next(
        (col for col in df.columns if any(keyword in col.lower() for keyword in time_keywords)), None
    )
    uploaded_data["time_column"] = time_column

    if time_column:
        time_display = f"Detected: {time_column}"
    else:
        time_display = "No time column detected"

    # Populate dropdown with column names
    y_options = [{"label": col, "value": col} for col in df.columns if col != time_column]
    return y_options, time_display


# Callback to update the graph based on dropdown selection
@app.callback(
    Output("main-graph", "figure"),
    Input("feature-dropdown", "value"),
)
def update_graph(selected_columns):
    global uploaded_data
    df = uploaded_data.get("df", None)
    time_column = uploaded_data.get("time_column", None)

    if df is None or time_column is None or not selected_columns:
        return go.Figure()

    fig = go.Figure()

    # Add traces for selected columns
    for column in selected_columns:
        fig.add_trace(
            go.Scatter(
                x=df[time_column],
                y=df[column],
                mode="lines+markers",
                name=column,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Electrolyzer Performance",
        xaxis_title=time_column,
        yaxis_title="Values",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
