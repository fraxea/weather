import matplotlib.pyplot as plt

def show_histogram(X, titles, row=2, column=3):
    fig, ax = plt.subplots(row, column, subplot_kw=dict(box_aspect=1))
    index_title = 0
    for i,ax_row in enumerate(ax):
        for j,axes in enumerate(ax_row):
            axes.set_title(titles[index_title])
            axes.hist(X[:, index_title], bins=365)
            index_title += 1
    fig.tight_layout()
    fig.set_size_inches(fig.get_size_inches() * 2)
