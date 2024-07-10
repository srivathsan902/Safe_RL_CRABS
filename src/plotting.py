from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication

class PlotWindow(QMainWindow):
    def __init__(self, window_title='Plot Window', x_label='Episodes', y_label='Reward', title_name='Reward vs Episodes'):
        super().__init__()
        self.x_label = x_label
        self.y_label = y_label
        self.window_title = window_title
        self.title_name = title_name
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.window_title)
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Initialize empty lists to store data
        self.x_data = []
        self.y_data = []

        # Timer to update plot every 1000 ms (1 second)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Update plot every 1 second

    def update_plot(self):
        # Clear axis and plot updated data
        self.ax.clear()
        self.ax.plot(self.x_data, self.y_data, marker='o', linestyle='-')
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title_name)

        # Update canvas
        self.canvas.draw()

    def add_data(self, x, y):
        self.x_data.append(x)
        self.y_data.append(y)
        self.update_plot()


def setup_plotting():
    app = QApplication(sys.argv)
    window_reward = PlotWindow(window_title='Rewards', x_label='Episode', y_label='Reward', title_name='Episode Reward')
    window_cost = PlotWindow(window_title='Costs', x_label='Episode', y_label='Cost', title_name='Episode Cost')
    window_reward.show()
    window_cost.show()
    return app, window_reward, window_cost