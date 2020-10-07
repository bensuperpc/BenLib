import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
#import matplotlib as mpl
import sys

pd.set_option('display.float_format', '{:.12f}'.format)
#mpl.rcParams['figure.dpi'] = 300

arg = sys.argv[1:]
df = None

if len(arg) == 0:
    df = pd.read_csv('log.csv', sep=",")
else:
    df = pd.read_csv(arg[0], sep=",")

print(df)
df = df.set_index("number")
df = df.sort_index()
print(df)
#ax = df.plot(marker='.')
ax = df.plot()

#ax.set_xticklabels(df.columns, rotation=0)
#ax.ticklabel_format(useOffset=False, style='plain')
##df.T

ax.set_yscale('log')
#ax.set_xscale('log')

ax.grid(color='b', linestyle='-', linewidth=0.1)
ax.set_xlabel('X elements')
ax.set_ylabel('Y time (s)')
ax.set_facecolor((1.0, 1.0, 1.0))

#start, end = ax.get_xlim()
#ax.xaxis.set_ticks(np.arange(start, end, 0.712123))
#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

#locx = ticker.MultipleLocator(base=2.0)
#ax.xaxis.set_major_locator(locx)

#def scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    #return '%.2E' % x
#    return '%.1E' % x

#scientific_formatter = ticker.FuncFormatter(scientific)
#ax.yaxis.set_major_formatter(scientific_formatter)

plt.title('Prime functions perf')

plt.show() 
fig = ax.get_figure()
fig.set_size_inches((21,9))
fig.savefig('graph.png', bbox_inches='tight', dpi=300)


#THANK https://stackoverflow.com/questions/33888973/get-values-from-matplotlib-axessubplot
#https://www.analyticsvidhya.com/blog/2020/05/10-matplotlib-tricks-data-visualization-python/
#https://stackoverflow.com/questions/6282058/writing-numerical-values-on-the-plot-with-matplotlib
#https://stackoverflow.com/questions/54165569/can-i-turn-of-scientific-notation-in-matplotlib-bar-chart
