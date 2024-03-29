from pyspark.sql import SparkSession

# Crea una sessione Spark
spark = SparkSession.builder \
    .appName("Verifica funzionamento Spark") \
    .getOrCreate()

try:
    # Crea un DataFrame di esempio
    data = [("Alice", 34), ("Bob", 45), ("Charlie", 37)]
    df = spark.createDataFrame(data, ["Name", "Age"])

    # Visualizza il DataFrame
    print("Contenuto del DataFrame:")
    df.show()

    # Esegui un'operazione di conteggio
    count = df.count()
    print(f"Numero di righe nel DataFrame: {count}")

    # Fai qualcos'altro con Spark...

finally:
    # Chiudi la sessione Spark
    spark.stop()
