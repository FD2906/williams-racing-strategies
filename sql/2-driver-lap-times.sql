-- Query written for Google BigQuery
-- Dataset: williams-racing-465223.f1_wc_1950_2024
-- Purpose: Query laptimes for Williams 2015–2019 drivers, labelling drivers as rookie or experienced.
-- Result stored in processed_data/driver-lap-times.csv

SELECT 
  L.raceId AS race_id,
  R.year AS gp_year,
  R.name AS gp_name,
  R.round AS gp_round,

  
  L.driverId AS driver_id,
  CONCAT(D.forename, ' ', D.surname) AS driver_name,
  CASE
    WHEN L.driverId IN (9, 13, 822) THEN 'experienced' -- kubica, massa, and bottas
    ELSE 'rookie' -- stroll, sirotkin, and russell
  END AS rookie_or_experienced, -- KPI 4

  L.lap AS lap_number,
  L.time AS lap_time,
  L.milliseconds AS lap_time_ms -- KPI 4

FROM williams-racing-465223.f1_wc_1950_2024.lap_times L
JOIN williams-racing-465223.f1_wc_1950_2024.drivers D 
  ON L.driverId = D.driverId
JOIN williams-racing-465223.f1_wc_1950_2024.races R 
  ON L.raceId = R.raceId
WHERE
  L.driverId IN (9, 13, 822, 840, 845, 847) -- the 6 williams drivers between 2015-2019
  AND R.year BETWEEN 2015 AND 2019
  AND L.milliseconds IS NOT NULL
ORDER BY R.year, R.round, driver_name, lap_number;