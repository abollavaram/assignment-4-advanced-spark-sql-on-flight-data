from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import unix_timestamp, col, abs, stddev, count, when, avg, hour, desc, rank, sum, broadcast, udf
from pyspark.sql.types import StringType, IntegerType

# Initialize Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load the datasets
flights = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports = spark.read.csv("airports.csv", header=True, inferSchema=True)

# Pre-processing to handle dates and time calculations
flights = flights.withColumn("ScheduledTravelTime", unix_timestamp("ScheduledArrival") - unix_timestamp("ScheduledDeparture")) \
                 .withColumn("ActualTravelTime", unix_timestamp("ActualArrival") - unix_timestamp("ActualDeparture")) \
                 .withColumn("DepartureDelay", unix_timestamp("ActualDeparture") - unix_timestamp("ScheduledDeparture"))

# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
flights = flights.withColumn("Discrepancy", abs(col("ScheduledTravelTime") - col("ActualTravelTime")))
windowSpec1 = Window.partitionBy("CarrierCode").orderBy(desc("Discrepancy"))
flights = flights.withColumn("Rank", rank().over(windowSpec1))
task1_result = flights.filter(col("Rank") <= 10) \
    .select("FlightNum", "CarrierCode", "Origin", "Destination", "ScheduledTravelTime", "ActualTravelTime", "Discrepancy")
task1_result.write.mode("overwrite").csv("output/task1_largest_discrepancy.csv", header=True)

# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
carrier_performance = flights.groupBy("CarrierCode") \
                             .agg(stddev("DepartureDelay").alias("StdDevDepartureDelay"), count("FlightNum").alias("TotalFlights")) \
                             .filter("TotalFlights > 100") \
                             .orderBy("StdDevDepartureDelay")
carrier_performance.write.mode("overwrite").csv("output/task2_consistent_airlines.csv", header=True)

# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
flights = flights.withColumn("Canceled", when(col("ActualDeparture").isNull(), 1).otherwise(0).cast(IntegerType()))
cancel_rates = flights.groupBy("Origin", "Destination").agg(
    (sum("Canceled") / count("*") * 100).alias("CancellationRate")
)

# Join with the airports table for origin airport details
cancel_rates = cancel_rates.join(
    broadcast(airports).alias("a1"), 
    cancel_rates["Origin"] == col("a1.AirportCode"), 
    "left"
).select(
    col("Origin"), 
    col("Destination"), 
    col("a1.AirportName").alias("OriginAirportName"), 
    col("a1.City").alias("OriginCity"), 
    col("CancellationRate")
)

# Join with the airports table for destination airport details
cancel_rates = cancel_rates.join(
    broadcast(airports).alias("a2"), 
    cancel_rates["Destination"] == col("a2.AirportCode"), 
    "left"
).select(
    col("Origin"),
    col("Destination"),
    col("OriginAirportName"),
    col("a2.AirportName").alias("DestinationAirportName"),
    col("OriginCity"),
    col("a2.City").alias("DestinationCity"),
    col("CancellationRate")
).orderBy(desc("CancellationRate"))

cancel_rates.write.mode("overwrite").csv("output/task3_canceled_routes.csv", header=True)

# Task 4: Carrier Performance Based on Time of Day
def time_of_day(dep_hour):
    if 5 <= dep_hour < 12:
        return "morning"
    elif 12 <= dep_hour < 17:
        return "afternoon"
    elif 17 <= dep_hour < 21:
        return "evening"
    else:
        return "night"

time_of_day_udf = udf(time_of_day, StringType())
flights = flights.withColumn("TimeOfDay", time_of_day_udf(hour("ScheduledDeparture")))
carrier_time_performance = flights.groupBy("CarrierCode", "TimeOfDay") \
                                  .agg(avg("DepartureDelay").alias("AvgDepartureDelay")) \
                                  .orderBy("CarrierCode", "TimeOfDay", "AvgDepartureDelay")
carrier_time_performance.write.mode("overwrite").csv("output/task4_carrier_performance_time_of_day.csv", header=True)

# Stop Spark session
spark.stop()
