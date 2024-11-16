# Databricks notebook source
# DBTITLE 1,Create table to hold extracted pdf text
# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS `dip-ragproject3`;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS `dip-ragproject3`.docs_text (
# MAGIC     id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC     text STRING
# MAGIC ) 
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# DBTITLE 1,Create table to track which pdf files we've already processed
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS `dip-ragproject3`.docs_track (file_name STRING) tblproperties (delta.enableChangeDataFeed = true);