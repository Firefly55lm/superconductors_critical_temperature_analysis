import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_sns_theme(theme_path: str, apply:bool = True) -> json:
    """
    Loads the Seaborn theme from a json file
    -----
    Args:
        * theme_path: path of the json file
        * apply: applies the theme to Seaborn if True
    --------
    Returns:
        * A json formatted file
    """

    with open(theme_path) as file:
        theme = json.load(file)
        file.close()
    
    if apply is True:
        sns.set_style("dark", rc=theme)
    
    return theme


if __name__ == "__main__":
    pass