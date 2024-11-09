from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    flights_df = flights_df.withColumn("scheduled_travel_time", F.col("ScheduledArrival") - F.col("ScheduledDeparture"))
    flights_df = flights_df.withColumn("actual_travel_time", F.col("ActualArrival") - F.col("ActualDeparture"))
    flights_df = flights_df.withColumn("discrepancy", F.abs(F.col("scheduled_travel_time") - F.col("actual_travel_time")))

    window_spec = Window.partitionBy("CarrierCode").orderBy(F.desc("discrepancy"))
    flights_df = flights_df.withColumn("Rank", F.row_number().over(window_spec))
    
    largest_discrepancy = flights_df.filter(F.col("Rank") == 1).select(
        "FlightNum", "CarrierCode", "Origin", "Destination",
        "scheduled_travel_time", "actual_travel_time", "discrepancy"
    ).join(carriers_df, "CarrierCode", "left").select(
        "FlightNum", "CarrierName", "Origin", "Destination", 
        "scheduled_travel_time", "actual_travel_time", "discrepancy"
    )

    largest_discrepancy.show()
    largest_discrepancy.write.csv(task1_output, header=True, mode="overwrite")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    flights_df = flights_df.withColumn(
        "departure_delay_minutes",
        (F.col("ActualDeparture").cast("long") - F.col("ScheduledDeparture").cast("long")) / 60
    )
    
    consistent_airlines = flights_df.groupBy("CarrierCode").agg(
        F.stddev("departure_delay_minutes").alias("stddev_departure_delay"),
        F.count("FlightNum").alias("flight_count")
    ).filter(F.col("flight_count") > 100).orderBy("stddev_departure_delay")
    
    consistent_airlines = consistent_airlines.withColumn(
        "rank", F.row_number().over(Window.orderBy("stddev_departure_delay"))
    ).join(carriers_df, "CarrierCode", "left").select(
        "rank", "CarrierName", "flight_count", "stddev_departure_delay"
    )

    consistent_airlines.show()
    consistent_airlines.write.csv(task2_output, header=True, mode="overwrite")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # Add a column to indicate if a flight was canceled (1 if canceled, 0 otherwise)
    flights_df = flights_df.withColumn(
        "IsCancelled", F.when(F.col("ActualDeparture").isNull(), 1).otherwise(0)
    )

    # Calculate the cancellation rate and total flights for each origin-destination pair
    canceled_routes = flights_df.groupBy("Origin", "Destination").agg(
        (F.avg("IsCancelled") * 100).alias("cancellation_rate_percentage"),  # Convert to percentage
        F.count("FlightNum").alias("total_flights")
    ).filter(F.col("total_flights") > 50).orderBy(F.desc("cancellation_rate_percentage"))

    # Alias the airports DataFrame for origin and destination joins
    origin_airports = airports_df.alias("origin_airport")
    destination_airports = airports_df.alias("destination_airport")

    # Join with the airports DataFrame to get names and cities for origin and destination
    canceled_routes = canceled_routes.join(
        origin_airports, canceled_routes["Origin"] == origin_airports["AirportCode"], "left"
    ).join(
        destination_airports, canceled_routes["Destination"] == destination_airports["AirportCode"], "left"
    ).select(
        F.col("origin_airport.AirportName").alias("Origin_AirportName"),
        F.col("origin_airport.City").alias("Origin_City"),
        F.col("destination_airport.AirportName").alias("Destination_AirportName"),
        F.col("destination_airport.City").alias("Destination_City"),
        "cancellation_rate_percentage"
    )

    # Add ranking based on cancellation rate percentage
    window_spec = Window.orderBy(F.desc("cancellation_rate_percentage"))
    canceled_routes_with_rank = canceled_routes.withColumn(
        "rank", F.row_number().over(window_spec)
    )

    # Show the DataFrame and write the result to CSV
    canceled_routes_with_rank.show()
    canceled_routes_with_rank.write.csv(task3_output, header=True, mode="overwrite")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    flights_df = flights_df.withColumn("DepartureHour", F.hour("ActualDeparture"))
    flights_df = flights_df.withColumn(
        "time_of_day",
        F.when((F.col("DepartureHour") >= 6) & (F.col("DepartureHour") < 12), "Morning")
         .when((F.col("DepartureHour") >= 12) & (F.col("DepartureHour") < 18), "Afternoon")
         .when((F.col("DepartureHour") >= 18) & (F.col("DepartureHour") < 24), "Evening")
         .otherwise("Night")
    )

    carrier_performance = flights_df.groupBy("CarrierCode", "time_of_day").agg(
        F.avg(F.col("ActualDeparture").cast("long") - F.col("ScheduledDeparture").cast("long")).alias("avg_departure_delay")
    )

    window_spec = Window.partitionBy("time_of_day").orderBy(F.desc("avg_departure_delay"))
    carrier_performance_with_rank = carrier_performance.withColumn(
        "rank", F.row_number().over(window_spec)
    ).join(
        carriers_df, "CarrierCode", "left"
    ).select(
        "CarrierCode", "CarrierName", "time_of_day", "avg_departure_delay", "rank"
    )

    carrier_performance_with_rank.show()
    carrier_performance_with_rank.write.csv(task4_output, header=True, mode="overwrite")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()