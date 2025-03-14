from flask import Flask, jsonify
from flask_cors import CORS
import logging
import traceback
from Database.RoadStateModel import RoadStateModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Road State API is running. Try /api/road-state/latest or /api/health"})

@app.route('/api/road-state/latest', methods=['GET'])
def get_latest_road_state():
    try:
        logger.debug("Attempting to get latest road state")
        road_state_model = RoadStateModel()
        latest_entry = road_state_model.get_latest()

        if not latest_entry:
            logger.info("No road state data available")
            return jsonify({"error": "No road state data available"}), 404

        logger.debug("Successfully retrieved road state data")
        return jsonify(latest_entry)
    except Exception as e:
        logger.error(f"Error getting latest road state: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

# For debugging directly
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000, debug=True)