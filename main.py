
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
import numpy as np
import matplotlib.pyplot as plt
from utils.stepfit import stepfit
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def main():
    # --- start GUI
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
