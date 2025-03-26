import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, date_sub, desc, lit
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import month, regexp_replace, row_number, to_date
from pyspark.sql.window import Window

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE_PATH = os.path.join(CURRENT_DIR, "Divvy_Trips_2019_Q4.csv")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "out")


# Function a: Calculate the average trip duration per day
def average_trip_duration_per_day(df):
    return df.groupBy("trip_date").agg(
        avg("tripduration_clean").alias("avg_trip_duration")
    )


# Function b: Count the number of trips per day
def trips_per_day(df):
    return df.groupBy("trip_date").agg(count("*").alias("trip_count"))


# Function c: Determine the most popular starting station for each month
def most_popular_start_station_per_month(df):
    station_counts = df.groupBy("trip_month", "from_station_name").agg(
        count("*").alias("trip_count")
    )
    windowSpec = Window.partitionBy("trip_month").orderBy(desc("trip_count"))
    return (
        station_counts.withColumn("rn", row_number().over(windowSpec))
        .filter(col("rn") == 1)
        .drop("rn")
    )


# Function d: Get the top 3 stations (by trip count) for each day for the last two weeks
def top3_stations_last_two_weeks(df):
    # Get the maximum trip_date in the dataset
    max_date = df.agg(spark_max("trip_date")).collect()[0][0]
    # Filter data for the last 14 days (including max_date)
    df_last2 = df.filter(col("trip_date") >= date_sub(lit(max_date), 13))
    station_counts = df_last2.groupBy("trip_date", "from_station_name").agg(
        count("*").alias("trip_count")
    )
    windowSpec = Window.partitionBy("trip_date").orderBy(desc("trip_count"))
    return (
        station_counts.withColumn("rn", row_number().over(windowSpec))
        .filter(col("rn") <= 3)
        .drop("rn")
    )


# Function e: Compare the average trip duration by gender
def average_duration_by_gender(df):
    # Filter out records with null or empty gender values
    df_gender = df.filter((col("gender").isNotNull()) & (col("gender") != ""))
    return df_gender.groupBy("gender").agg(
        avg("tripduration_clean").alias("avg_trip_duration")
    )


def main() -> None:
    spark = SparkSession.builder.appName("BikeTripsAnalysis").getOrCreate()

    df = spark.read.option("header", True).csv(INPUT_FILE_PATH)

    # Data cleansing:
    # 1. Remove commas from the 'tripduration' field and cast it to double
    # 2. Extract the trip date and month from the 'start_time' column
    df = df.withColumn(
        "tripduration_clean",
        regexp_replace(col("tripduration"), ",", "").cast("double"),
    )
    df = df.withColumn("trip_date", to_date(col("start_time"), "yyyy-MM-dd HH:mm:ss"))
    df = df.withColumn("trip_month", month(col("start_time")))

    # a. Average trip duration per day, ordered by date
    avg_duration_df = average_trip_duration_per_day(df).orderBy("trip_date")
    avg_duration_df.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "average_trip_duration_per_day")
    )

    # b. Number of trips per day, ordered by date
    trips_per_day_df = trips_per_day(df).orderBy("trip_date")
    trips_per_day_df.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "trips_per_day")
    )

    # c. Most popular starting station per month, ordered by month
    popular_station_df = most_popular_start_station_per_month(df).orderBy("trip_month")
    popular_station_df.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "most_popular_start_station_per_month")
    )

    # d. Top 3 stations for each day for the last two weeks, ordered by date and descending trip count
    top3_stations_df = top3_stations_last_two_weeks(df).orderBy(
        "trip_date", desc("trip_count")
    )
    top3_stations_df.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "top3_stations_last_two_weeks")
    )

    # e. Average trip duration by gender, ordered by descending average duration
    gender_avg_df = average_duration_by_gender(df).orderBy(desc("avg_trip_duration"))
    gender_avg_df.write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_PATH, "average_duration_by_gender")
    )

    spark.stop()


if __name__ == "__main__":
    main()
