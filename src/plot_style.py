"""Report typography: Times New Roman Latin, SimSun text, SimHei headings."""
import matplotlib as mpl


def configure_fonts():
    mpl.rcParams.update({'font.family': ['Times New Roman', 'SimSun'],
                         'font.serif': ['Times New Roman', 'SimSun'],
                         'font.sans-serif': ['Times New Roman', 'SimSun'],
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def finish_fonts(fig):
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        text.set_fontfamily(['Times New Roman', 'SimSun'])
    for ax in fig.axes:
        ax.title.set_fontfamily(['Times New Roman', 'SimHei'])
    if fig._suptitle:
        fig._suptitle.set_fontfamily(['Times New Roman', 'SimHei'])
