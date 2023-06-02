DROP DATABASE IF EXISTS RETAIL;
CREATE DATABASE RETAIL;
USE RETAIL;

DROP TABLE IF EXISTS sales_positions;

CREATE TABLE sales_positions (
  sales_id    BIGINT,
  position_id INT,
  article_id  INT,
  amount      INT,
  price       DOUBLE,
  voucher_id  INT,
  canceled    BOOLEAN
)
STORED AS PARQUET
TBLPROPERTIES ("parquet.compression"="SNAPPY");

INSERT INTO TABLE sales_positions
VALUES (1, 1, 1, 10, 10.0, 1, true), (2,2,2,20,20.0,2,false);
