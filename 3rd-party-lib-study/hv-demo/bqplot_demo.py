from bqplot import *
from bqplot import pyplot as plt
size = 100
x_data = range(size)
np.random.seed(0)
y_data = np.cumsum(np.random.randn(size) * 100.0)
y_data_2 = np.cumsum(np.random.randn(size))
y_data_3 = np.cumsum(np.random.randn(size) * 100.)

sc_ord = OrdinalScale()
sc_y = LinearScale()
sc_y_2 = LinearScale()

ord_ax = Axis(label='Test X', scale=sc_ord, tick_format='0.0f', grid_lines='none')
y_ax = Axis(label='Test Y', scale=sc_y, 
            orientation='vertical', tick_format='0.2f', 
            grid_lines='solid')
y_ax_2 = Axis(label='Test Y 2', scale=sc_y_2, 
              orientation='vertical', side='right', 
              tick_format='0.0f', grid_lines='solid')

line_chart = Lines(x=x_data[:10], y = [y_data[:10], y_data_2[:10] * 100, y_data_3[:10]],
                   scales={'x': sc_ord, 'y': sc_y},
                   labels=['Line1', 'Line2', 'Line3'], 
                   display_legend=True)

bar_chart = Bars(x=x_data[:10], 
                 y=[y_data[:10], y_data_2[:10] * 100, y_data_3[:10]], 
                 scales={'x': sc_ord, 'y': sc_y_2},
                 labels=['Bar1', 'Bar2', 'Bar3'],
                 display_legend=True)

# the line does not have a Y value set. So only the bars will be displayed
Figure(axes=[ord_ax, y_ax],  marks=[bar_chart, line_chart], legend_location = 'bottom-left')