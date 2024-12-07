CREATE TABLE rooms (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE cameras (
    id TEXT PRIMARY KEY,
    room_id TEXT REFERENCES rooms(id),
    name TEXT,
    capabilities JSON,
    last_seen TIMESTAMP,
    status TEXT,
    FOREIGN KEY (room_id) REFERENCES rooms(id)
);

CREATE TABLE alarm ( 
    id TEXT PRIMARY KEY,
    room_id TEXT REFERENCES rooms(id),
    camera_id TEXT REFERENCES cameras(id),
    timestamp TIMESTAMP,
    FOREIGN KEY (room_id) REFERENCES rooms(id),
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
)

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    password TEXT NOT NULL
);

CREATE TABLE firm (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
)