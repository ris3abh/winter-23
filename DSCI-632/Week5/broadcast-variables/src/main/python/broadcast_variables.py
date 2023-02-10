#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
from pyspark.sql import SparkSession

"""
    Uses a broadcast variable to generate summary information
    about a list of store locations.

    camelCase used for variable names for consistency and
    ease of translation across the prevailing style of
    the Java, R, and Scala examples.
"""
if __name__ == "__main__":
    spark = SparkSession.builder.appName("broadcast_variables").getOrCreate()

    # Register state data and schema as broadcast variables
    localDF = spark.read.json("us_states.json")
    broadcastStateData = spark.sparkContext.broadcast(localDF.collect())
    broadcastSchema = spark.sparkContext.broadcast(localDF.schema)

    # Create a DataFrame based on the store locations.
    storesDF = spark.read.json("store_locations.json")

    # Create a DataFrame of US state data with the broadcast variables.
    stateDF = spark.createDataFrame(broadcastStateData.value, broadcastSchema.value)

    # Join the DataFrames to get an aggregate count of stores in each US Region
    print("How many stores are in each US region?")
    joinedDF = storesDF.join(stateDF, "state").groupBy("census_region").count()
    joinedDF.show()

    spark.stop()
