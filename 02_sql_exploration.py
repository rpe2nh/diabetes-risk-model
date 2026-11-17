import duckdb

con = duckdb.connect()
con.sql("CREATE VIEW brfss AS SELECT * FROM read_csv_auto('data/diabetes_brfss2015.csv')")

print("1. Overall diabetes rate")
con.sql("""
    SELECT COUNT(*) AS people,
           ROUND(AVG(Diabetes_binary) * 100, 1) AS pct_diabetes
    FROM brfss
""").show()

print("2. Diabetes rate by age group (1 = 18-24 ... 13 = 80+)")
con.sql("""
    SELECT Age AS age_group,
           COUNT(*) AS people,
           ROUND(AVG(Diabetes_binary) * 100, 1) AS pct_diabetes
    FROM brfss
    GROUP BY Age
    ORDER BY Age
""").show()

print("3. Diabetes rate by BMI category")
con.sql("""
    SELECT CASE
             WHEN BMI < 18.5 THEN 'Underweight'
             WHEN BMI < 25   THEN 'Healthy weight'
             WHEN BMI < 30   THEN 'Overweight'
             ELSE 'Obese'
           END AS bmi_category,
           COUNT(*) AS people,
           ROUND(AVG(Diabetes_binary) * 100, 1) AS pct_diabetes
    FROM brfss
    GROUP BY bmi_category
    ORDER BY MIN(BMI)
""").show()

print("4. Diabetes rate by high blood pressure and high cholesterol")
con.sql("""
    SELECT HighBP, HighChol,
           COUNT(*) AS people,
           ROUND(AVG(Diabetes_binary) * 100, 1) AS pct_diabetes
    FROM brfss
    GROUP BY HighBP, HighChol
    ORDER BY HighBP, HighChol
""").show()

print("5. Diabetes rate by self-rated health (1 = excellent ... 5 = poor)")
con.sql("""
    SELECT GenHlth,
           COUNT(*) AS people,
           ROUND(AVG(Diabetes_binary) * 100, 1) AS pct_diabetes
    FROM brfss
    GROUP BY GenHlth
    ORDER BY GenHlth
""").show()