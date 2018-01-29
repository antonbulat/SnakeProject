import plotly.plotly as py
import plotly.graph_objs as go
import plotly
#plotly.tools.set_credentials_file(username='jsiekier', api_key='3NLogCbIsa0w056N3OlK')
x = open("learning_results_small_new.txt")
myEdge = []
x_axis=[]
y_axis_avg=[]
y_axis_max=[]
for t in x.readlines():
    sp = t.split(";")
    scores= sp[0].split(",")
    int_scores=[float(sc) for sc in scores]
    x_axis.append(int(sp[1])*4)
    y_axis_avg.append(sum(int_scores)/len(int_scores))
    y_axis_max.append(max(int_scores))


trace1 = go.Scatter(
    x = x_axis,
    y = y_axis_avg,
    mode = 'lines+markers',
    name = 'avg'
)
trace2 = go.Scatter(
    x=x_axis,
    y=y_axis_max,
    mode = 'lines+markers',
    name = 'max'
)
data = [trace1, trace2]
# Edit the layout
layout = dict(title = '3x same action, small size',
              xaxis = dict(title = 'time in minutes'),
              yaxis = dict(title = 'number of apples'),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='small-size-24h')