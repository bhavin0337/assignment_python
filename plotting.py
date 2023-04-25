from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, grid
from bokeh.models import Band, ColumnDataSource

def func_plot_ideal(ideal_functions, file_name):
    
    ideal_functions.sort(key=lambda ideal_function: ideal_function.func_train.name, reverse=False)
    plots = []
    for ideal_function in ideal_functions:
        p = two_func_for_graph_plot(line_function=ideal_function, scatter_function=ideal_function.func_train,
                                          squared_error=ideal_function.error)
        plots.append(p)
    output_file("{}.html".format(file_name))
    
    show(column(*plots))

def func_ideal_with_points_plot(points_with_classification, file_name):
    
    plots = []
    for index, item in enumerate(points_with_classification):
        if item["classification"] is not None:
            p = class_plot(item["point"], item["classification"])
            plots.append(p)
    output_file("{}.html".format(file_name))
    show(column(*plots))

def two_func_for_graph_plot(scatter_function, line_function, squared_error):
   
    datafr_f1 = scatter_function.dataframe
    naam_f1 = scatter_function.name

    datafr_f2 = line_function.dataframe
    naam_f2 = line_function.name

    squared_error = round(squared_error, 2)
    p = figure(title="train model {} vs ideal {}. Total squared error = {}".format(naam_f1, naam_f2, squared_error),
               x_axis_label='x', y_axis_label='y')
    p.scatter(datafr_f1["x"], datafr_f1["y"], fill_color="red", legend_label="Train")
    p.line(datafr_f2["x"], datafr_f2["y"], legend_label="Ideal", line_width=2)
    return p

def class_plot(point, ideal_function):
    
    if ideal_function is not None:
        datafr_func_class = ideal_function.dataframe

        str_po = "({},{})".format(point["x"], round(point["y"], 2))
        title = "point {} with classification: {}".format(str_po, ideal_function.name)

        p = figure(title=title, x_axis_label='x', y_axis_label='y')

        
        p.line(datafr_func_class["x"], datafr_func_class["y"],
                legend_label="Classification function", line_width=2, line_color='black')

        criterion = ideal_function.tolerance
        datafr_func_class['upper'] = datafr_func_class['y'] + criterion
        datafr_func_class['lower'] = datafr_func_class['y'] - criterion

        source = ColumnDataSource(datafr_func_class.reset_index())

        band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay',
            fill_alpha=0.3, line_width=1, line_color='green', fill_color="green")

        p.add_layout(band)
     
        p.scatter([point["x"]], [round(point["y"], 4)], fill_color="red", legend_label="Test point", size=8)

        return p
