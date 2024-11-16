# Databricks notebook source
# DBTITLE 1,Create table to hold extracted pdf text
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS llm.rag.docs_text ( id BIGINT GENERATED BY DEFAULT AS IDENTITY, text STRING ) tblproperties (delta.enableChangeDataFeed = true);

# COMMAND ----------

# DBTITLE 1,Create table to track which pdf files we've already processed
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS llm.rag.docs_track (file_name STRING) tblproperties (delta.enableChangeDataFeed = true);