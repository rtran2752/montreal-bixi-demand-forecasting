-- Typical demand by weekday and hour
SELECT CAST(strftime('%w', datetime) AS INTEGER) AS weekday,
       CAST(strftime('%H', datetime) AS INTEGER) AS hour,
       ROUND(AVG(demand), 1) AS avg_departures
FROM hourly_demand
GROUP BY weekday, hour
ORDER BY weekday, hour;

-- Highest-demand stations during the evening commute
SELECT station_id, SUM(departures) AS departures
FROM station_hourly
WHERE CAST(strftime('%H', datetime) AS INTEGER) BETWEEN 16 AND 18
GROUP BY station_id
ORDER BY departures DESC
LIMIT 20;

