from base64 import b64encode

import io
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def plot_empty_data(ticker, title_override=None):
    # Create an empty plot
    title = title_override or f"No data available for {ticker}"
    fig = plt.figure(figsize=(12, 2))
    plt.plot([], [])
    # Set labels and title
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # Show the plot
    return fig

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.tight_layout()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return b64encode(img.read()).decode('utf8')