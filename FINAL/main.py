from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import StandardScaler, VectorAssembler, PCA
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pandas as pd
import numpy as np
import json

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D



class ClusterHandler():
    """
    Runs and handles a Spark session.
    -----
    Args:
        * name [optional]: the name of the session.
        * type [optional]: 'local' or remote session on a masternode. If 'remote', must provide IPv4 address and port number. 'local' is default.
        * ip [optional]: IPv4 address of the node.
        * port [optional]: port number of the node.
    -------
    Raises:
        * ValueError: if type is not 'local' or 'remote'.
    """

    def __init__(self, name:str = 'Session', type:str = 'local', ip:str = None, port:str = None, config:str = None):
        self.session = self.run_session(name=name, type=type, ip=ip, port=port, config=config)
        self.data = None
        self.dataframe = None


    def run_session(self, name:str = 'Session', type:str = 'local', ip:str = None, port:str = None, config:str = None) -> SparkSession.sparkContext:
        """
        Runs the spark session.
        -----
        Args:
            * name [optional]: the name of the session.
            * type [optional]: 'local' or remote session. If 'remote', must provide IPv4 address and port number. 'local' is default.
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

        self.context = spark.sparkContext
        print(f"\nSession '{self.context.appName}' created on masternode {self.context.master}")
        print(f"Spark UI (jobs tab) is available at \033[36m{self.context.uiWebUrl}\033[0m\n")
        
        return spark


    def generate_dataframe(self, data:pd.DataFrame, y:str = None):
        """
        Generates a Spark Dataframe on the spark session from the given Pandas DataFrame.
        -----
        Args:
            * data: the data source as Pandas Dataframe.
            * y [optional]: the name of the y column if you need to drop it from the main dataset (suggested for PCA).
        --------
        Returns:
            * self.data: the original dataframe.
            * self.dataframe: the Spark ataframe.
        """
        self.data = data
        if y:
            self.y = pd.DataFrame(data["critical_temp"])
            X = data.drop(columns=["critical_temp"])
            self.dataframe = self.session.createDataFrame(X)
        else:
            self.dataframe = self.session.createDataFrame(data)

    
    def assemble_features(self, assembler:VectorAssembler = None, input_columns:list = None, output_column:str = "features"):
        """
        Runs an assembler on the session's dataset.
        -----
        Args:
            * assembler [optional]: assembler to use. If not provided, a default one will be used.
            * input_columns [optional]: columns to assemble as list of strings.
            * output_column [optional]: name of the output column as string.
        """
        if not assembler:
            if not input_columns:
                input_columns = list(self.dataframe.columns[0:])
            assembler = VectorAssembler(inputCols=input_columns, outputCol=output_column)
        self.dataframe = assembler.transform(self.dataframe)


    def scale_features(self, scaler:StandardScaler = None, input_column:str = "features", output_column:str = "scaledFeatures"):
        """
        Runs a scaler on the session's dataset.
        -----
        Args:
            * scaler [optional]: scaler to use. If not provided, a default one will be used.
            * input_column [optional]: the name of the features column previously assembled as string. "features" is default.
            * output_column [optional]: name of the output column as string. "
        """
        if not scaler:
            scaler = StandardScaler(inputCol=input_column, outputCol=output_column, withStd=True, withMean=True)
        self.dataframe = scaler.fit(self.dataframe).transform(self.dataframe)


    def fit_pca(self, model:PCA):
        """
        Fits a PCA (Principal Component Analysis) model. The output returned will be available as Class Attribute.
        -----
        Args:
            * model: a Spark PCA model.
        --------
        Returns:
            * self.pca_model: the fitted PCA model.
            * self.pca_result: the transformed dataset with a PCA dedicated column.
            * self.pca_coefficients: the features' coefficients calculated by the PCA model.
        """
        self.pca_model = model.fit(self.dataframe)
        self.pca_result = self.pca_model.transform(self.dataframe)
        self.pca_coefficients = self.pca_model.pc.toArray()

        pca_result_sub = self.pca_result.select("pcaFeatures").collect()
        pca_values = [tuple(row.pcaFeatures.toArray()) for row in pca_result_sub]
        self.pca_transposed = list(zip(*pca_values))
        

    def extract_pca_coefficients(self, dimension:int) -> dict:
        """
        Pairs the PCA coefficients with the feature's name for a given dimension.
        -----
        Args:
            * dimension: the dimension of the coefficients of interest.
        --------
        Returns:
            * coefficients: a dictionary with feature_name:coefficient as key:value pairs.
        """
        feat_coeff = {feature: coefficient[dimension] for feature, coefficient in zip(self.dataframe.columns, self.pca_coefficients)}
        return dict(sorted(feat_coeff.items(), key=lambda x: abs(x[1]), reverse=True))


    def fit_lr(self, model:LinearRegression, y_column:str, pred_column:str = "prediction", folds:int = 5,
               reg_params:list = [0.001, 0.01, 0.1], elasticnet_params:list = [0, 0.5, 1]) -> dict:
        """
        Fits a LR (Linear Regression) model, tuning the reg_param parameter with a k-fold Cross Validation. The metric is R2.
        -----
        Args:
            * model: a Spark LinearRegression model.
            * y_column: name of the target y variable as string.
            * pred_column [optional]: the name of the assembled features column as string. "prediction" is default.
            * folds [optional]: number of folds to perform the k-fold Cross Validation. 5 is default.
            * reg_params [optional]: a list of values for the regularization params (lambda) to tune the model. [0.01, 0.05, 1.0] are default.
            * elasticnet_params [optional]: a list of values for the regularization behavior. 0 is ridge (L2), 1 is lasso (L1), 0.5 is a combination.
        --------
        Returns:
            * CV results: a dictionary with reg_param:metric as key:value pairs.
            * self.lr_model: the fitted LR model.
            * self.lr_coefficients: the features' coefficients calculated by the LR model.
        """
        
        param_grid = ParamGridBuilder() \
                    .addGrid(model.regParam, reg_params) \
                    .addGrid(model.elasticNetParam, elasticnet_params) \
                    .build()
        
        evaluator = RegressionEvaluator(predictionCol=pred_column, labelCol=y_column, metricName="r2")
        cross_validator = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=folds)

        cv_model = cross_validator.fit(self.dataframe)
        self.lr_model = cv_model.bestModel
        self.lr_coefficients = list(self.lr_model.coefficients)
        param_maps = cv_model.getEstimatorParamMaps()

        return {f"Lambda: {list(params.values())[0]}; Elasticnet param: {list(params.values())[1]}": r2 for params, r2 in zip(param_maps, cv_model.avgMetrics)}
        

    def extract_lr_coefficients(self) -> dict:
        """
        Pairs the LR coefficients with the feature's name.
        --------
        Returns:
            * coefficients: a dictionary with feature_name:coefficient as key:value pairs.
        """
        feat_coeff = {feature: coefficient for feature, coefficient in zip(self.dataframe.columns, self.lr_coefficients)}
        return dict(sorted(feat_coeff.items(), key=lambda x: abs(x[1]), reverse=True))
    

    def plot_3d_pca(self, dimensions=list, color_by:str = None, cmap:str = "rocket"):
        """
        Plots a 3D chart for the given PCA dimensions using MatPlotLib.
        -----
        Args:
            * dimensions: the 3 dimensions indexes as list of integers (0 is the first).
            * color_by: a variable to dye the scatter's dots. A legend will be created.
            * cmap: the colormap to use.
        """
        if len(dimensions) != 3:
            raise ValueError("\033[31mMust provide 3 dimensions to plot.\033[0m")

        pca_transposed = self.pca_transposed

        self.load_theme("dark")
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(f'Dimension {dimensions[0]}')
        ax.set_ylabel(f'Dimension {dimensions[1]}')
        ax.set_zlabel(f'Dimension {dimensions[2]}')
        ax.set_title(f'3D PCA scatter for dimensions {dimensions[0]}, {dimensions[1]}, {dimensions[2]}', color="white")

        if color_by:
            color_feature = self.data[color_by].to_numpy()
            ax.scatter(xs=pca_transposed[dimensions[0]], ys=pca_transposed[dimensions[1]], zs=pca_transposed[dimensions[2]],
                       c=color_feature, cmap=cmap, norm=Normalize(vmin=0, vmax=100), s=2, alpha=0.6)
            
            color_map = matplotlib.colormaps.get_cmap(cmap)
            legend_elements = [
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=mcolors.to_hex(color_map(0.1)), markersize=10, label='Low'),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=mcolors.to_hex(color_map(0.5)), markersize=10, label='Medium'),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=mcolors.to_hex(color_map(0.9)), markersize=10, label='High')
                            ]

            legend = ax.legend(handles=legend_elements, title=color_by, loc='upper left', labels=['Low', 'Medium', 'High'])
            for text in legend.get_texts():
                text.set_color('white')
            legend.get_title().set_color('white')
            legend.get_title().set_weight('bold')
        
        else:
            ax.scatter(xs=pca_transposed[dimensions[0]], ys=pca_transposed[dimensions[1]], zs=pca_transposed[dimensions[2]],
                       s=2, alpha=0.6, color="orange")
        plt.show()


    def plot_2d_pca(self, dimensions=list, color_by:str = None, cmap:str = "rocket"):
        """
        Plots a 2D chart for the given PCA dimensions using Seaborn.
        -----
        Args:
            * dimensions: the 2 dimensions indexes as list of integers (0 is the first).
            * color_by: a variable to dye the scatter's dots. A legend will be created.
            * cmap: the colormap to use.
        """
        if len(dimensions) != 2:
            raise ValueError("\033[31mMust provide 2 dimensions to plot.\033[0m")

        self.load_theme("light")
        pca_transposed = self.pca_transposed

        plt.xlabel(f'Dimension {dimensions[0]}')
        plt.ylabel(f'Dimension {dimensions[1]}')
        plt.title(f'2D PCA scatter for dimensions {dimensions[0]}, {dimensions[1]}', color="black")

        if color_by:
            color_feature = self.data[color_by].to_numpy()
            sns.scatterplot(x=pca_transposed[dimensions[0]], y=pca_transposed[dimensions[1]],
                       c=color_feature, cmap=cmap, norm=Normalize(vmin=0, vmax=100), s=2, alpha=0.6)
            
            color_map = matplotlib.colormaps.get_cmap(cmap)
            legend_elements = [
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=mcolors.to_hex(color_map(0.1)), markersize=10, label='Low'),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=mcolors.to_hex(color_map(0.5)), markersize=10, label='Medium'),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=mcolors.to_hex(color_map(0.9)), markersize=10, label='High')
                            ]

            legend = plt.legend(handles=legend_elements, title=color_by, loc='upper left', labels=['Low', 'Medium', 'High'])
            for text in legend.get_texts():
                text.set_color('white')
            legend.get_title().set_color('white')
            legend.get_title().set_weight('bold')
        
        else:
            sns.scatterplot(x=pca_transposed[dimensions[0]], y=pca_transposed[dimensions[1]],
                       s=2, alpha=0.6, color="orange")
        plt.show()


    def load_theme(self, theme_name:str) -> dict:
        """
        Loads the Seaborn/Matplolib theme.
        -----
        Args:
            * theme_name: the name of the theme ("dark" or "light")
        --------
        Returns:
            * A dictionary with the theme's information.
        """

        dark = {
            "figure.facecolor": "#202021",
            "axes.facecolor": "#262626",
            "axes.edgecolor": "#cfcfd1",
            "axes.grid": True,
            "grid.color": "#555555",
            "grid.linewidth": 0.5,
            "xtick.color": "#ffffff",
            "ytick.color": "#ffffff",
            "axes.labelcolor": "#ffffff"
            }
        
        light = {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#171717",
            "axes.grid": True,
            "grid.color": "#555555",
            "grid.linewidth": 0.5,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.labelcolor": "#000000"
            }
        
        if theme_name == "dark":
            sns.set_style("dark", rc=dark)
            return dark
        elif theme_name == "light":
            sns.set_style("dark", rc=light)
            return light
        else:
            raise ValueError("\033[31mPlease provide a valid theme name.\033[0m")



if __name__ == "__main__":
        pass
    