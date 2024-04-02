from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.feature import PCA

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from utilities.utils import load_sns_theme


class ClusterHandler():

    def __init__(self, data:pd.DataFrame, y:str = None):
        self.data = data
        if y:
            self.y = pd.DataFrame(data[y])
            self.X = data.drop(columns=[y])
        else:
            self.X = data
        self.dataframe = None
        self.pca_model = None
        self.pca_result = None


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


    def generate_dataframe(self):
        self.dataframe = self.session.createDataFrame(self.X)

    
    def assemble_features(self, assembler:VectorAssembler = None):
        if not assembler:
            assembler = VectorAssembler(inputCols=list(self.dataframe.columns[0:]), outputCol="features")
        self.dataframe = assembler.transform(self.dataframe)


    def scale_features(self, scaler:StandardScaler = None, ):
        if not scaler:
            scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
        self.dataframe = scaler.fit(self.dataframe).transform(self.dataframe)


    def fit_pca(self, model:PCA):
        self.pca_model = model.fit(self.dataframe)
        self.pca_result = self.pca_model.transform(self.dataframe)


    def extract_pca_coefficients(self, dimension:int):
        pca_coefficients = self.pca_model.pc.toArray()
        feat_coeff = {feature: coefficient[dimension] for feature, coefficient in zip(self.data.columns, pca_coefficients)}
        return dict(sorted(feat_coeff.items(), key=lambda x: abs(x[1]), reverse=True))
    

    def plot_3d_pca(self, dimensions=list, color_by:str = None):
        
        try:
            load_sns_theme(r"utilities\themes\fire_theme.json")
        except:
            raise Exception("\nCannot load any theme. Please, use load_sns_theme from the utils library to load from a json.")

        pca_result_sub = self.pca_result.select("pcaFeatures").collect()
        pca_values = [tuple(row.pcaFeatures.toArray()) for row in pca_result_sub]
        pca_transposed = list(zip(*pca_values))

        if len(dimensions) != 3:
            raise ValueError("\033[31mYou must provide 3 dimensions to plot.\033[0m")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(f'Dimension {dimensions[0]}')
        ax.set_ylabel(f'Dimension {dimensions[1]}')
        ax.set_zlabel(f'Dimension {dimensions[2]}')
        ax.set_title(f'3D PCA scatter for dimensions {dimensions[0]}, {dimensions[1]}, {dimensions[2]}', color="white")

        if color_by:
            color_feature = self.data[color_by].to_numpy()
            ax.scatter(xs=pca_transposed[dimensions[0]], ys=pca_transposed[dimensions[1]], zs=pca_transposed[dimensions[2]],
                       c=color_feature, cmap="rocket", s=2, alpha=0.6)
            
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#f5c5ac', markersize=10, label='Low'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#c6004e', markersize=10, label='Medium'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#251432', markersize=10, label='High')
            ]

        
            legend = ax.legend(handles=legend_elements, title=color_by, loc='upper left', labels=['Low', 'Medium', 'High'])
            for text in legend.get_texts():
                text.set_color('white')
            legend.get_title().set_color('white')
            legend.get_title().set_weight('bold')
        
        else:
            ax.scatter(xs=pca_transposed[dimensions[0]], ys=pca_transposed[dimensions[1]], zs=pca_transposed[dimensions[2]],
                       s=2, alpha=0.6, color="orange")



if __name__ == "__main__":

    data = pd.read_csv("data/superconductivity.csv")
    y = pd.DataFrame(data["critical_temp"])
    X = data.drop(columns=["critical_temp"])
    
    handler = ClusterHandler(X)
    handler.run_session()
    handler.generate_dataframe()

    handler.assemble_features()
    handler.scale_features()

    pca = PCA(k=5, inputCol="scaledFeatures", outputCol="pcaFeatures")
    pca_result = handler.fit_pca(pca)
    pca_result.show()

    print(handler.extract_pca_coefficients(model=pca, dimension=0))



    
    handler.session.stop()