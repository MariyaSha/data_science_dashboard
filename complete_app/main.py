# Stock Data Dashboard Application
# Author: Mariya Sha @ Python Simplified

# GUI imports
import taipy as tp
import taipy.gui.builder as tgb
from taipy.gui import Icon
from taipy import Config
# machine learning imports
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras import models
from tensorflow.keras import layers
# data imports
import datetime
import plotly.graph_objects as go
import os
try:
    import cudf
    import cudf.pandas as pd
    print("Using cuDF for GPU acceleration.")
except ImportError:
    import pandas as pd
    print("Using pandas on CPU.")


###################################
# GLOBAL VARIABLES
###################################

# datasets
stock_data = pd.read_csv("data/sp500_stocks.csv")
company_data = pd.read_csv("data/sp500_companies.csv")

# contry names and icons [for slider]
country_names = company_data["Country"].unique().tolist()
country_names = [(i, Icon("images/flags/" + i + ".png", i)) for i in country_names]

# company names [for slider]
company_names = company_data[["Symbol", "Shortname"]].sort_values("Shortname").values.tolist()

# start and finish dates
dates = [
    stock_data["Date"].min(),
    stock_data["Date"].max()
]

# initial country and company selection
country = "Canada"
company = ["LULU"]

# initial prediction values
lin_pred = 0
knn_pred = 0
rnn_pred = 0

# initial graph values
graph_data = None
figure = None

###################################
# GRAPHIC USER INTERFACE
###################################

# create web page
with tgb.Page() as page:
    # create horizontal group of elements
    # aligned to the center
    with tgb.part("text-center"):
        tgb.image("images/icons/logo.png", width="10vw")
        tgb.text(
            "# S&P 500 Stock Value Over Time",
            mode="md"
            )
        # create date range selector
        tgb.date_range(
            "{dates}",
            label_start="Start Date",
            label_end="End Date"
            )
        # create vertical group of 2 elements
        # taking 20% and 80% of the view poer
        with tgb.layout("20 80"):
            tgb.selector(
                label="country",
                class_name="fullwidth",
                value="{country}",
                lov="{country_names}",
                dropdown=True,
                value_by_id=True
                )
            tgb.selector(
                label="company",
                class_name="fullwidth",
                value="{company}",
                lov="{company_names}",
                dropdown=True,
                value_by_id=True,
                multiple=True
                )
        # create chart
        tgb.chart(figure="{figure}")
        # vertical group of 8 elements
        with tgb.part("text-left"):
            with tgb.layout("4 72 4 4 4 4 4 4"):
                # company name and symbol
                tgb.image(
                    "images/icons/id-card.png",
                    width="3vw"
                    )
                tgb.text("{company[-1]} | {company_data['Shortname'][company_data['Symbol'] == company[-1]].values[0]}", mode="md")
                # linear regression prediction
                tgb.image(
                    "images/icons/lin.png",
                    width="3vw"
                    )
                tgb.text("{lin_pred}", mode="md")
                # KNN prediction
                tgb.image(
                    "images/icons/knn.png",
                    width="3vw"
                    )
                tgb.text("{knn_pred}", mode="md")
                # RNN prediction
                tgb.image(
                    "images/icons/rnn.png",
                    width="3vw"
                    )
                tgb.text("{rnn_pred}", mode="md")
                
###################################
# FUNCTIONS
###################################

def build_company_names(country):
    """
    filter companies by their country of origin
    
    - input country: string with country name
    - output company_names: list of company names from the input country
    """
    company_names = company_data[["Symbol", "Shortname"]][
        company_data["Country"] == country
    ].sort_values("Shortname").values.tolist()
    
    return company_names

def build_graph_data(dates, company):
    """
    filter global stock data by dates and companies
    
    - input dates: list with a start date and a finish date
    - input company: list of company symbols
    
    - output graph_data: numpy array of stock values
      for each symbol within the date range
    """
    temp_data = stock_data[["Date", "Adj Close", "Symbol"]][
        # filter by dates
        (stock_data["Date"] > str(dates[0])) &
        (stock_data["Date"] < str(dates[1]))
    ]
    
    # reconstruct temp_data with empty data frame
    graph_data = pd.DataFrame()
    # fetch dates column
    graph_data["Date"] = temp_data["Date"].unique()
    
    # fetch company values into new columns
    for i in company:
        graph_data[i] = temp_data["Adj Close"][
            temp_data["Symbol"] == i
        ].values

    return graph_data

def display_graph(graph_data):
    """
    draw stock value graphs
    - input graph_data: numpy array of stock values to plot
    - output figure: plotly Figure with visuallized graph_data
    """
    figure = go.Figure()
    # fetch symbols from column names
    symbols = graph_data.columns[1:]
    
    # draw historic data for each symbol
    for i in symbols:
        figure.add_trace(go.Scatter(
            x=graph_data["Date"],
            y=graph_data[i],
            name=i,
            showlegend=True
            ))
        
    # add titles
    figure.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Value"
        )
    
    return figure

def split_data(stock_data, dates, symbol):
    """
    arrange data for training and prediction
    
    - input stock_data: Pandas dataframe from global variable
    - input dates: list with a start date and a finish date
    - input symbol: string that represents a company symbol
    
    - output features: numpy array
    - output targets: numpy array
    - output eval_features: numpy array
    """
    temp_data = stock_data[
        # filter dates and symbol
        (stock_data["Symbol"] == symbol) &
        (stock_data["Date"] > str(dates[0])) &
        (stock_data["Date"] < str(dates[1]))
    ].drop(["Date", "Symbol"], axis=1)
    
    # fetch evaluation sample
    eval_features = temp_data.values[-1]
    # unsqueeze dimensions
    eval_features = eval_features.reshape(1, -1)
    # fetch features and targets
    features = temp_data.values[:-1]
    targets = temp_data["Adj Close"].iloc[1:].values
    
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    # normalize features
    features -= mean
    features /= std
    # normalize evaluation sample
    eval_features -= mean
    eval_features /= std

    return features, targets, eval_features

def get_lin(dates, company):
    """
    obtain prediction with Linear Regression
    
    - input dates: list with a start date and a finish date
    - input company: list of company symbols
    
    - output: floating point prediciton
    """
    x, y, eval_x = split_data(stock_data, dates, company[-1])

    lin_model.fit(x, y)
    lin_pred = lin_model.predict(eval_x)

    return round(lin_pred[0], 3)

def get_knn(dates, company):
    """
    obtain prediction with K Nearest Neighbors
    
    - input dates: list with a start date and a finish date
    - input company: list of company symbols
    
    - output: floating point prediciton
    """
    x, y, eval_x = split_data(stock_data, dates, company[-1])
    
    knn_model.fit(x, y)
    knn_pred = knn_model.predict(eval_x)

    return round(knn_pred[0], 3)

def get_rnn(dates, company):
    """
    obtain prediction with RNN
    
    - input dates: list with a start date and a finish date
    - input company: list of company symbols
    
    - output: floating point prediciton
    """
    x, y, eval_x = split_data(stock_data, dates, company[-1])
    
    rnn_model.fit(x, y, batch_size=32, epochs=10, verbose=0)
    rnn_pred = rnn_model.predict(eval_x)

    return round(float(rnn_pred[0][0]), 3)

###################################
# BACKEND
###################################

# configure data nodes
country_cfg = Config.configure_data_node(
    id="country"
)
company_names_cfg = Config.configure_data_node(
    id="company_names"
)
dates_cfg = Config.configure_data_node(
    id="dates"
)
company_cfg = Config.configure_data_node(
    id="company"
)
graph_data_cfg = Config.configure_data_node(
    id="graph_data"
)
lin_pred_cfg = Config.configure_data_node(
    id="lin_pred"
)
knn_pred_cfg = Config.configure_data_node(
    id="knn_pred"
)
rnn_pred_cfg = Config.configure_data_node(
    id="rnn_pred"
)

# configure tasks
get_lin_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = lin_pred_cfg,
    function = get_lin,
    id = "get_lin",
    skippable = True
    )

get_knn_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = knn_pred_cfg,
    function = get_knn,
    id = "get_knn",
    skippable = True
    )

get_rnn_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = rnn_pred_cfg,
    function = get_rnn,
    id = "get_rnn",
    skippable = True
    )

build_graph_data_cfg = Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = graph_data_cfg,
    function = build_graph_data,
    id = "build_graph_data",
    skippable = True
    )

build_company_names_cfg = Config.configure_task(
    input = country_cfg,
    output = company_names_cfg,
    function = build_company_names,
    id = "build_company_names",
    skippable = True
    )

# configure scenario
scenario_cfg = Config.configure_scenario(
    task_configs = [
        build_company_names_cfg, 
        build_graph_data_cfg,
        get_lin_cfg,
        get_knn_cfg,
        get_rnn_cfg
    ],
    id="scenario"
    )

def on_init(state):
    """
    built-in Taipy function that runs once
    when the application first loads
    """
    # write inputs to scenario
    state.scenario.country.write(state.country)
    state.scenario.dates.write(state.dates)
    state.scenario.company.write(state.company)
    # update scenario
    state.scenario.submit(wait=True)
    # fetch updated outputs
    state.graph_data = state.scenario.graph_data.read()
    state.company_names = state.scenario.company_names.read()
    state.lin_pred = state.scenario.lin_pred.read()
    state.knn_pred = state.scenario.knn_pred.read()
    state.rnn_pred = state.scenario.rnn_pred.read()

def on_change(state, name, value):
    """
    built-in Taipy function that runs every time
    a GUI variable is changed by user
    """
    if name == "country":
        print(name, "was modified", value)
        # update scenario with new country selection
        state.scenario.country.write(state.country)
        state.scenario.submit(wait=True)
        state.company_names = state.scenario.company_names.read()
    
    if name == "company" or name == "dates":
        print(name, "was modified", value)
        # update scenario with new company or dates selection
        state.scenario.dates.write(state.dates)
        state.scenario.company.write(state.company)
        state.scenario.submit(wait=True)
        state.graph_data = state.scenario.graph_data.read()
        state.lin_pred = state.scenario.lin_pred.read()
        state.knn_pred = state.scenario.knn_pred.read()
        state.rnn_pred = state.scenario.rnn_pred.read()
    
    if name == "graph_data":
        # display updated graph data
        state.figure = display_graph(state.graph_data)

def build_RNN(n_features):
    """
    create a Recurrent Neural Network
    - input n_fratures: integer with the number of features within x and eval_x
    - output model: RNN Tensorflow model
    """
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(n_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model

if __name__ == "__main__":
    # create machine learning models
    lin_model = LinearRegression()
    knn_model = KNeighborsRegressor()
    rnn_model = build_RNN(6)
    # run Taipy orchestrator to manage scenarios
    tp.Orchestrator().run()
    # intialize scenario
    scenario = tp.create_scenario(scenario_cfg)
    # initialize GUI and display page
    gui = tp.Gui(page)
    # run application
    gui.run(
        title = "Data Science Dashboard",
        # automatically reload app when main.py is saved
        use_reloader = True
        )
