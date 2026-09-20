CREATE TABLE IF NOT EXISTS hourly_demand (
    datetime TIMESTAMP PRIMARY KEY,
    demand INTEGER NOT NULL CHECK (demand >= 0),
    temp_c REAL,
    relative_humidity REAL,
    wind_speed_kmh REAL,
    precip_mm REAL
);

CREATE TABLE IF NOT EXISTS stations (
    station_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    capacity INTEGER
);

CREATE TABLE IF NOT EXISTS station_hourly (
    datetime TIMESTAMP NOT NULL,
    station_id TEXT NOT NULL,
    departures INTEGER NOT NULL CHECK (departures >= 0),
    PRIMARY KEY (datetime, station_id),
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

CREATE INDEX IF NOT EXISTS ix_station_hourly_datetime ON station_hourly(datetime);

