import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql import Row, functions as F
from pyspark.sql.window import Window


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    This function builds a spark session
    """
    spark = SparkSession.builder.appName("Sparkify-ETL").config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0").getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    This function pulls song_data data from an S3 bucket. It then extracts and processes the data for the songs and artists tables. Finally storing the data to S3 in parquet format.

    Parameters:
    spark - A spark session must be created
    input_data - The source data S3 location
    output_data - The S3 location of the processed data    
    """
    
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data', '*', '*', '*')
    
    # read song data file
    df = spark.read.json(song_data)
    
    # extract columns to create songs table
    songs_table = df.select(['song_id', 'title', 'year','duration']).dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year','title').mode('overwrite').parquet(output_data+'songs/songs.parquet')

    # extract columns to create artists table
    artists_table = df.select(['artist_id', 'artist_name', 'artist_location', 'artist_latitude','artist_longitude']) \
            .withColumnRenamed('artist_name', 'name') \
            .withColumnRenamed('artist_location', 'location') \
            .withColumnRenamed('artist_latitude', 'latitude') \
            .withColumnRenamed('artist_longitude', 'longitude') \
            .dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(output_data+'artists/artists.parquet')


def process_log_data(spark, input_data, output_data):
    """
    This function pulls log_data data from an S3 bucket. It then extracts and processes the data for the users and times tables. Then it will combine the song and the log data to create the songplays table, finally storing the data to S3 in parquet format.

    Parameters:
    spark - A spark session must be created
    input_data - The source data S3 location
    output_data - The S3 location of the processed data    
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data','*','*')

    # read log data file
    log_df = spark.read.json(log_data)
      
    # filter by actions for song plays
    log_df = log_df.filter(log_df['page'] == 'NextSong') 
            

    # extract columns for users table 
    users_table = log_df.select(['userId', 'firstName', 'lastName', 'gender', 'level', \
                             F.row_number().over(Window.partitionBy('userId').orderBy(F.desc('ts'))).alias('rowNum')]) 
    users_table = users_table.select(['userId', 'firstName', 'lastName', 'gender', 'level']) \
                            .filter(users_table.rowNum == 1)     
    
    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data+'users/users.parquet')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    log_df = log_df.withColumn('timestamp', get_timestamp(log_df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000)))
    log_df = log_df.withColumn('start_time', get_datetime(log_df.ts))
    
    # extract columns to create time table
    # start_time, hour, day, week, month, year, weekday
    time_table = log_df.select('start_time') \
                    .withColumn('hour', F.hour('start_time')) \
                    .withColumn('day', F.dayofmonth('start_time')) \
                    .withColumn('week', F.weekofyear('start_time')) \
                    .withColumn('month', F.month('start_time')) \
                    .withColumn('year', F.year('start_time')) \
                    .withColumn('weekday', F.when(F.dayofweek('start_time') < 5, 1).otherwise(0)) \
                    .dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year','month').mode('overwrite').parquet(output_data+'time/time.parquet')

    # read in song data to use for songplays table
    song_data = os.path.join(input_data, 'song_data', '*', '*', '*')
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table 
    # songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
    log_song_table = log_df.join(song_df, (log_df.song == song_df.title)\
                                 & (log_df.artist == song_df.artist_name) \
                                 & (log_df.length == song_df.duration), how='inner')
    
    songplays_table = log_song_table.select([F.monotonically_increasing_id().alias('songplay_id'),'start_time', 'userId','level','song_id','artist_id', 'sessionId', 'location', 'userAgent']) \
                                    .withColumn('month', F.month('start_time')) \
                                    .withColumn('year', F.year('start_time')) \
                                    .withColumnRenamed('userId', 'user_id') \
                                    .withColumnRenamed('userID', 'user_id') \
                                    .withColumnRenamed('userAgent', 'user_agent') \
                                    .withColumnRenamed('sessionId', 'session_id')

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year','month').mode('overwrite').parquet(output_data+'songplays/songplays.parquet')


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://cy-udacity-storage/sparkify/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)
    spark.stop()

if __name__ == "__main__":
    main()