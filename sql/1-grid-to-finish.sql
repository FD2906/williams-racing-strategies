-- Query written for Google BigQuery
-- Dataset: williams-racing-465223.f1_wc_1950_2024
-- Purpose: Calculate grid-to-finish position delta for Williams & midfield drivers (2015–2019)
-- Result stored in processed_data/grid-to-finish.csv

SELECT
  RA.raceId AS race_id,
  RA.year AS gp_year,
  RA.name AS gp_name,
  RA.round AS gp_round,
  CONCAT(D.forename, ' ', D.surname) AS driver_name,
  C.name AS constructor,
  C.constructorRef AS constructor_ref, -- constructor reference
  C.constructorRef = 'williams' AS is_williams,-- for filtering out williams later
  CAST(RE.grid AS INT64) AS start_position, -- start position
  CAST(RE.position AS INT64) AS final_position, -- final position
  CAST(RE.grid AS INT64) - CAST(RE.position AS INT64) AS grid_delta -- KPI 1
FROM 
  williams-racing-465223.f1_wc_1950_2024.results RE -- results.csv
JOIN williams-racing-465223.f1_wc_1950_2024.drivers D -- drivers.csv
  ON RE.driverId = D.driverId
JOIN williams-racing-465223.f1_wc_1950_2024.constructors C -- constructors.csv
  ON RE.constructorId = C.constructorId
JOIN williams-racing-465223.f1_wc_1950_2024.races RA -- races.csv
  ON RE.raceId = RA.raceId
WHERE 
  (RA.year BETWEEN 2015 AND 2019) -- time constraint
  AND (C.constructorRef IN ('williams', 'renault', 'haas', 'force_india', 'racing_point')) -- midfield team comparison
  AND (RA.name IN (
    'Italian Grand Prix', 
    'Monaco Grand Prix', 
    'British Grand Prix', 
    'Belgian Grand Prix',
    'Spanish Grand Prix',
    'Singapore Grand Prix',
    'Brazilian Grand Prix',
    'Hungarian Grand Prix',
    'Austrian Grand Prix',
    'Japanese Grand Prix')) -- filter to most consistent 10 circuits across 2015-2019
  AND (RE.position IS NOT NULL) -- removes NULL position finishes
  AND RE.grid > 0 -- pit lane starts not included.
  