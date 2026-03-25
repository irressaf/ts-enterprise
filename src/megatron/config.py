import pandas as pd
from datetime import datetime

SEASONAL_PERIOD = 1
MIN_LENGTH = 7
COUNTRY = "RU"
MIN_DATE = pd.to_datetime("1900-01-01")
MAX_DATE = datetime.today()
FH_SIZE = 1
SEED = None
FIG_WIDTH = 900
FIG_HEIGHT = 500
COLOR = "#1f77b4"
MARGIN = dict(l=30, r=30, t=50, b=30)


def set_config(**kwargs):
    globals().update(kwargs)
