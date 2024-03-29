from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import ElementwiseProduct

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from utilities.utils import load_sns_theme


class ClusterHandler():

    def __init__(self):
        self.session = None
        

    def run_session(self, name:str = 'Session', type:str = 'local', ip:str = None, port:str = None, config:str = None) -> SparkSession.sparkContext:
        """
        Runs the spark session
        -----
        Args:
            * name: the name of the session.
            * type: 'local' or remote session. If 'remote', must provide IPv4 address and port number. 'local' is default.
            * ip [optional]: IPv4 address of the node.
            * port [optional]: port number of the node.
        -------
        Raises:
            * ValueError: if type is not 'local' or 'remote'.
        """

        type = type.lower()
        
        if type == "local":
            spark = SparkSession.builder \
                .appName(name) \
                .getOrCreate()
        
        elif type == "remote":
            if ip == None or port == None:
                raise ValueError("\033[31mMust provide 'ip' and 'port' in order to connect to remote node.\033[0m")
            
            spark = SparkSession.builder \
                .appName(name) \
                .master(f"spark://{ip}:{port}") \
                .getOrCreate()
            
        else:
            raise ValueError("\033[31mArgument 'type' must be 'local' or 'remote'.\033[0m")

        self.session = spark
        context = spark.sparkContext
        print(f"\nSession '{context.appName}' created on masternode {context.master}")
        print(f"Spark UI is available at \033[36m{context.uiWebUrl}\033[0m\n")
        
        return context

    def assemble_features(self, assembler:VectorAssembler = None):
        feature_cols = list(df_spark.columns[0:])
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")





if __name__ == "__main__":
    
    spark = ClusterHandler()
    spark.run_session()

    df = pd.read_csv("data/superconductivity.csv")
    df_spark = spark.session.createDataFrame(df)
    
    spark.session.stop()