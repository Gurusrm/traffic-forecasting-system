# API Reference

## Endpoints

### `GET /`
Returns the dashboard HTML.

### `GET /locations`
Returns a list of sensor locations.
- **Response**: `[{"id": "773869", "lat": 34.15, "lng": -118.32}, ...]`

### `GET /sample`
Fetches a random test sample.
- **Response**: `{"input": [...], "target": [...], "start_index": 123}`

### `POST /predict`
Runs inference on the provided input.
- **Body**: `{"input_data": [...]}` (Shape: `[12, 207, 2]`)
- **Response**: `{"prediction": [...]}` (Shape: `[1, 12, 207, 1]`)
